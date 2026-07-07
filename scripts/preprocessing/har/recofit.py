#!/usr/bin/env python3
"""Rebuild Recofit from the raw multi-activity MATLAB release.

This script follows the dataset audit playbook:
- process one raw right-arm visit at a time;
- low-pass filter at 10 Hz, then resample signals to 20 Hz;
- align labels by nearest neighbor on the 20 Hz timeline;
- preserve the existing processed filename contract and embedded-label layout.
"""

import os

# Cap BLAS/OpenMP threads before importing numpy/scipy so 24 workers do not oversubscribe.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm


DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./data/raw/Recofit"))
RAW_PATH = DATASET_ROOT / "raw" / "exercise_data.50.0000_multionly.mat"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "processed_rebuild_tmp"
TARGET_FS = 20.0
FILTER_ORDER = 4
FILTER_CUTOFF_HZ = 10.0
PLACEMENT = "rightarm"

LABEL_NORMALIZATION = {
    "Tap Left Device": "Tap IMU Device",
    "Tap Right Device": "Tap IMU Device",
}

RAW_SUBJECT_DATA = None
OUTPUT_DIR: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the audited Recofit processed subset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for rebuilt parquet files.",
    )
    parser.add_argument(
        "--recordings",
        nargs="*",
        default=None,
        help="Optional subset of visit tokens, e.g. s000v001 s045v000.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=24,
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


def visit_token(subject_idx: int, visit_idx: int) -> str:
    return f"s{subject_idx:03d}v{visit_idx:03d}"


def iter_recording_keys(subject_data: np.ndarray) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    for subject_idx, item in enumerate(subject_data):
        records = list(item.flat) if isinstance(item, np.ndarray) else [item]
        for visit_idx, _record in enumerate(records):
            keys.append((subject_idx, visit_idx))
    return keys


def get_recording(subject_idx: int, visit_idx: int):
    item = RAW_SUBJECT_DATA[subject_idx]
    if isinstance(item, np.ndarray):
        return item.flat[visit_idx]
    if visit_idx != 0:
        raise IndexError(f"{visit_token(subject_idx, visit_idx)} is not valid for a single-visit subject.")
    return item


def effective_fs(times_s: np.ndarray) -> float:
    diffs = np.diff(times_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if len(diffs) == 0:
        raise ValueError("Cannot estimate sampling rate from fewer than two timestamps.")
    return float(1.0 / np.median(diffs))


def lowpass_if_needed(values: np.ndarray, *, fs: float, cutoff: float, order: int) -> np.ndarray:
    if fs <= TARGET_FS or len(values) <= (order * 3 + 1):
        return values

    nyq = 0.5 * fs
    cutoff_eff = min(cutoff, np.nextafter(nyq, 0.0))
    if cutoff_eff <= 0.0:
        return values

    b, a = butter(order, cutoff_eff / nyq, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))
    if len(values) <= padlen:
        return values
    return filtfilt(b, a, values, axis=0)


def nearest_labels(raw_times: np.ndarray, raw_labels: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(raw_times, target_times, side="left")
    idx = np.clip(idx, 0, len(raw_times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(raw_times) - 1)
    use_prev = np.abs(target_times - raw_times[prev_idx]) <= np.abs(raw_times[idx] - target_times)
    nearest_idx = np.where(use_prev, prev_idx, idx)
    return raw_labels[nearest_idx]


def interpolate_missing(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    if np.isfinite(values).all():
        return values

    repaired = values.astype(np.float64, copy=True)
    for axis_idx in range(repaired.shape[1]):
        axis_values = repaired[:, axis_idx]
        valid = np.isfinite(axis_values)
        if valid.sum() < 2:
            raise ValueError("Signal axis has fewer than two finite samples.")
        repaired[:, axis_idx] = np.interp(times_s, times_s[valid], axis_values[valid])
    return repaired


def fill_missing_labels(raw_labels: np.ndarray) -> np.ndarray:
    missing = raw_labels == ""
    if not missing.any():
        return raw_labels

    valid_idx = np.flatnonzero(~missing)
    if len(valid_idx) == 0:
        raise ValueError("Recording has no labeled samples.")

    idx = np.arange(len(raw_labels), dtype=np.int64)
    prev_idx = np.where(~missing, idx, -1)
    np.maximum.accumulate(prev_idx, out=prev_idx)

    next_idx = np.where(~missing, idx, len(raw_labels))
    next_idx = np.minimum.accumulate(next_idx[::-1])[::-1]

    filled = raw_labels.copy()
    miss_idx = np.flatnonzero(missing)
    use_prev = np.abs(miss_idx - prev_idx[miss_idx]) <= np.abs(next_idx[miss_idx] - miss_idx)
    chosen = np.where(use_prev, prev_idx[miss_idx], next_idx[miss_idx])
    filled[miss_idx] = raw_labels[chosen]
    return filled


def build_raw_labels(raw_times: np.ndarray, activity_start_matrix: np.ndarray) -> np.ndarray:
    raw_labels = np.full(len(raw_times), "", dtype=object)
    rows = np.asarray(activity_start_matrix, dtype=object)
    for row in rows:
        label = LABEL_NORMALIZATION.get(str(row[0]), str(row[0]))
        start_s = float(row[1])
        end_s = float(row[2])
        mask = (raw_times >= start_s) & (raw_times <= end_s)
        raw_labels[mask] = label
    return fill_missing_labels(raw_labels).astype(str, copy=False)


def build_target_times(raw_times: np.ndarray) -> np.ndarray:
    last_time_s = float(raw_times[-1])
    n_out = int(np.floor(last_time_s * TARGET_FS + 1e-9)) + 1
    return np.arange(n_out, dtype=np.float64) / TARGET_FS


def resample_signal(raw_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    fs = effective_fs(raw_times)
    repaired = interpolate_missing(values, raw_times)
    filtered = lowpass_if_needed(repaired, fs=fs, cutoff=FILTER_CUTOFF_HZ, order=FILTER_ORDER)

    rounded_fs = int(round(fs))
    expected_len = len(target_times)
    if fs > TARGET_FS and rounded_fs > 0 and abs(fs - rounded_fs) <= 0.05:
        resampled = resample_poly(filtered, up=int(round(TARGET_FS)), down=rounded_fs, axis=0)
        if len(resampled) > expected_len:
            resampled = resampled[:expected_len]
        elif 0 < len(resampled) < expected_len:
            pad = np.repeat(resampled[-1:, :], expected_len - len(resampled), axis=0)
            resampled = np.vstack([resampled, pad])
    else:
        resampled = np.column_stack(
            [np.interp(target_times, raw_times, filtered[:, axis_idx]) for axis_idx in range(filtered.shape[1])]
        )

    if len(resampled) != expected_len:
        raise ValueError(f"Expected {expected_len} samples after resampling, got {len(resampled)}.")
    return resampled.astype(np.float32, copy=False)


def signal_filename(sensor_name: str, target_times: np.ndarray, subject_idx: int, visit_idx: int) -> str:
    duration_token = int(round(float(target_times[-1]))) if len(target_times) else 0
    token = visit_token(subject_idx, visit_idx)
    return f"Recofit_{sensor_name}_{PLACEMENT}_{duration_token}_{token}.parquet"


def trim_to_common_overlap(acc_t: np.ndarray, acc_v: np.ndarray, gyro_t: np.ndarray, gyro_v: np.ndarray):
    overlap_start = max(float(acc_t[0]), float(gyro_t[0]))
    overlap_end = min(float(acc_t[-1]), float(gyro_t[-1]))
    if overlap_end <= overlap_start:
        raise ValueError("Accelerometer and gyroscope streams do not overlap.")

    acc_mask = (acc_t >= overlap_start) & (acc_t <= overlap_end)
    gyro_mask = (gyro_t >= overlap_start) & (gyro_t <= overlap_end)
    acc_t = acc_t[acc_mask] - overlap_start
    gyro_t = gyro_t[gyro_mask] - overlap_start
    acc_v = acc_v[acc_mask]
    gyro_v = gyro_v[gyro_mask]
    return acc_t, acc_v, gyro_t, gyro_v, overlap_start, overlap_end


def build_recording_data(subject_idx: int, visit_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    record = get_recording(subject_idx, visit_idx)

    acc_matrix = np.asarray(record.data.accelDataMatrix, dtype=np.float64)
    gyro_matrix = np.asarray(record.data.gyroDataMatrix, dtype=np.float64)
    acc_times, acc_values = acc_matrix[:, 0], acc_matrix[:, 1:4]
    gyro_times, gyro_values = gyro_matrix[:, 0], gyro_matrix[:, 1:4]
    acc_times, acc_values, gyro_times, gyro_values, overlap_start, overlap_end = trim_to_common_overlap(
        acc_times,
        acc_values,
        gyro_times,
        gyro_values,
    )

    label_times = np.asarray(record.data.accelDataMatrix[:, 0], dtype=np.float64)
    label_mask = (label_times >= overlap_start) & (label_times <= overlap_end)
    label_times = label_times[label_mask] - overlap_start
    raw_labels = build_raw_labels(np.asarray(record.data.accelDataMatrix[:, 0], dtype=np.float64), record.activityStartMatrix)
    raw_labels = raw_labels[label_mask]

    target_times = build_target_times(acc_times)
    aligned_labels = nearest_labels(label_times, raw_labels, target_times).astype(str, copy=False)
    acc_resampled = resample_signal(acc_times, acc_values, target_times)
    gyro_resampled = resample_signal(gyro_times, gyro_values, target_times)

    acc_df = pd.DataFrame(
        {
            "time_s": target_times,
            "x": acc_resampled[:, 0],
            "y": acc_resampled[:, 1],
            "z": acc_resampled[:, 2],
            "label": aligned_labels,
        }
    )
    gyro_df = pd.DataFrame(
        {
            "time_s": target_times,
            "x": gyro_resampled[:, 0],
            "y": gyro_resampled[:, 1],
            "z": gyro_resampled[:, 2],
            "label": aligned_labels,
        }
    )
    return acc_df, gyro_df


def process_recording(key: tuple[int, int]) -> dict[str, object]:
    subject_idx, visit_idx = key
    token = visit_token(subject_idx, visit_idx)
    acc_df, gyro_df = build_recording_data(subject_idx, visit_idx)

    acc_name = signal_filename("Accelerometer", acc_df["time_s"].to_numpy(dtype=np.float64), subject_idx, visit_idx)
    gyro_name = signal_filename("Gyroscope", gyro_df["time_s"].to_numpy(dtype=np.float64), subject_idx, visit_idx)
    acc_path = OUTPUT_DIR / acc_name
    gyro_path = OUTPUT_DIR / gyro_name
    acc_df.to_parquet(acc_path, index=False)
    gyro_df.to_parquet(gyro_path, index=False)

    counts = Counter(acc_df["label"].tolist())
    return {
        "token": token,
        "acc_file": acc_name,
        "gyro_file": gyro_name,
        "rows": int(len(acc_df)),
        "duration_s": float(acc_df["time_s"].iloc[-1]) if len(acc_df) else 0.0,
        "label_counts": dict(counts),
    }


def normalize_requested_tokens(tokens: Iterable[str]) -> set[str]:
    cleaned = set()
    for token in tokens:
        value = str(token).strip()
        if not value:
            continue
        if not value.startswith("s"):
            raise ValueError(f"Recording token must look like s000v000, got {value!r}.")
        cleaned.add(value)
    return cleaned


def main() -> None:
    global RAW_SUBJECT_DATA, OUTPUT_DIR

    args = parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    if OUTPUT_DIR.exists():
        if not args.overwrite:
            raise FileExistsError(f"{OUTPUT_DIR} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mat = loadmat(RAW_PATH, squeeze_me=True, struct_as_record=False)
    RAW_SUBJECT_DATA = mat["subject_data"]
    keys = iter_recording_keys(RAW_SUBJECT_DATA)

    if args.recordings:
        wanted = normalize_requested_tokens(args.recordings)
        keys = [key for key in keys if visit_token(*key) in wanted]
        missing = sorted(wanted - {visit_token(*key) for key in keys})
        if missing:
            raise ValueError(f"Unknown recording token(s): {missing}")

    if not keys:
        raise ValueError("No recordings selected.")

    ctx = mp.get_context("fork")
    results: list[dict[str, object]] = []
    with ctx.Pool(processes=int(args.n_jobs)) as pool:
        iterator = pool.imap_unordered(process_recording, keys, chunksize=1)
        for row in tqdm(iterator, total=len(keys), desc="Recofit rebuild"):
            results.append(row)

    results = sorted(results, key=lambda row: str(row["token"]))
    manifest_rows: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    for row in results:
        label_counts.update({str(k): int(v) for k, v in row["label_counts"].items()})
        manifest_rows.append(
            {
                "recording_token": row["token"],
                "acc_file": row["acc_file"],
                "gyro_file": row["gyro_file"],
                "rows": row["rows"],
                "duration_s": row["duration_s"],
            }
        )

    pd.DataFrame(manifest_rows).to_csv(OUTPUT_DIR / "rebuild_manifest.csv", index=False)
    pd.DataFrame(
        [{"label": label, "count": int(count)} for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    ).to_csv(OUTPUT_DIR / "label_counts.csv", index=False)

    print(f"Wrote {len(results) * 2} parquet files to {OUTPUT_DIR}")
    print(f"Rebuilt {len(results)} right-arm visits.")
    print("Top labels:", dict(label_counts.most_common(15)))


if __name__ == "__main__":
    main()
