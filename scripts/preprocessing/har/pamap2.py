#!/usr/bin/env python3
"""Rebuild PAMAP2 subject recordings for downstream evaluation.

This follows `scripts/preprocessing/DATASET_AUDIT_PLAYBOOK.md`:
- one output unit per raw subject `.dat` recording;
- preserve the existing PAMAP2 filename contract;
- low-pass filter each signal at 10 Hz before 20 Hz resampling;
- align labels by nearest timestamp, never by Fourier/linear label resampling;
- trim every subject to the maximum common overlap across all in-scope
  placement/sensor streams before writing shared labels.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


DATASET_NAME = "PAMAP2"
FOLDER_NAME = "PAMAP2_Dataset"
AXES = ("x", "y", "z")
PLACEMENTS = ("hand", "chest", "ankle")
SENSORS = ("acc1", "acc2", "gyro", "mag")

RAW_COLUMN_NAMES = [
    "timestamp_s",
    "activity_id",
    "heart_rate",
    "hand_temp",
    "hand_acc1_x",
    "hand_acc1_y",
    "hand_acc1_z",
    "hand_acc2_x",
    "hand_acc2_y",
    "hand_acc2_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orient_0",
    "hand_orient_1",
    "hand_orient_2",
    "hand_orient_3",
    "chest_temp",
    "chest_acc1_x",
    "chest_acc1_y",
    "chest_acc1_z",
    "chest_acc2_x",
    "chest_acc2_y",
    "chest_acc2_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orient_0",
    "chest_orient_1",
    "chest_orient_2",
    "chest_orient_3",
    "ankle_temp",
    "ankle_acc1_x",
    "ankle_acc1_y",
    "ankle_acc1_z",
    "ankle_acc2_x",
    "ankle_acc2_y",
    "ankle_acc2_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orient_0",
    "ankle_orient_1",
    "ankle_orient_2",
    "ankle_orient_3",
]

SIGNAL_COLUMNS = {
    ("hand", "acc1"): ("hand_acc1_x", "hand_acc1_y", "hand_acc1_z"),
    ("hand", "acc2"): ("hand_acc2_x", "hand_acc2_y", "hand_acc2_z"),
    ("hand", "gyro"): ("hand_gyro_x", "hand_gyro_y", "hand_gyro_z"),
    ("hand", "mag"): ("hand_mag_x", "hand_mag_y", "hand_mag_z"),
    ("chest", "acc1"): ("chest_acc1_x", "chest_acc1_y", "chest_acc1_z"),
    ("chest", "acc2"): ("chest_acc2_x", "chest_acc2_y", "chest_acc2_z"),
    ("chest", "gyro"): ("chest_gyro_x", "chest_gyro_y", "chest_gyro_z"),
    ("chest", "mag"): ("chest_mag_x", "chest_mag_y", "chest_mag_z"),
    ("ankle", "acc1"): ("ankle_acc1_x", "ankle_acc1_y", "ankle_acc1_z"),
    ("ankle", "acc2"): ("ankle_acc2_x", "ankle_acc2_y", "ankle_acc2_z"),
    ("ankle", "gyro"): ("ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"),
    ("ankle", "mag"): ("ankle_mag_x", "ankle_mag_y", "ankle_mag_z"),
}

LABEL_NAMES = {
    0: "other",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    24: "rope_jumping",
}

SUMMARY_COLUMNS = ["subject_id", "sensor"]
for placement in PLACEMENTS:
    for sensor in SENSORS:
        for axis in AXES:
            prefix = f"{placement}_{sensor}_{axis}"
            for stat in ("mean", "std", "min", "max", "range"):
                SUMMARY_COLUMNS.append(f"{prefix}_{stat}")


def parse_args() -> argparse.Namespace:
    dataset_root = Path(os.environ.get("DATASET_ROOT", "./data/raw/PAMAP2_Dataset"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=dataset_root / "dataset",
        help="Directory containing raw PAMAP2 subject .dat files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_root / "processed_rebuild_tmp",
        help="Directory to write rebuilt parquet artifacts.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subset of subject IDs to rebuild (e.g. subject101 subject109).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of worker processes for the per-subject rebuild.",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=20.0,
        help="Target sampling rate in Hz.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Low-pass cutoff in Hz before downsampling.",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Butterworth filter order.",
    )
    parser.add_argument(
        "--gap-threshold-s",
        type=float,
        default=1.0,
        help="Start a new segment when consecutive timestamps differ by more than this many seconds.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="pamap2_rebuild_summary.csv",
        help="Per-subject rebuild summary filename written inside the output directory.",
    )
    parser.add_argument(
        "--label-counts-csv",
        type=str,
        default="pamap2_label_counts.csv",
        help="Global rebuilt label-count CSV filename written inside the output directory.",
    )
    return parser.parse_args()


def normalize_subjects(subjects: list[str] | None) -> set[str] | None:
    if not subjects:
        return None
    return {subject.strip().lower() for subject in subjects if subject.strip()}


def split_at_gaps(times_s: np.ndarray, gap_threshold_s: float) -> list[slice]:
    if len(times_s) == 0:
        return []
    gaps = np.flatnonzero(np.diff(times_s) > gap_threshold_s) + 1
    starts = np.r_[0, gaps]
    ends = np.r_[gaps, len(times_s)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def effective_fs(times_s: np.ndarray) -> float:
    if len(times_s) < 2:
        return 0.0
    diffs = np.diff(times_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 0.0
    return float(1.0 / np.median(diffs))


def lowpass_if_needed(values: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    if fs <= 20.0 or len(values) <= (order * 3 + 1):
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


def interpolate_missing(times_s: np.ndarray, values: np.ndarray) -> np.ndarray | None:
    if np.isfinite(values).all():
        return values

    repaired = values.astype(np.float64, copy=True)
    for axis_idx in range(repaired.shape[1]):
        axis_values = repaired[:, axis_idx]
        valid = np.isfinite(axis_values)
        if valid.sum() < 2:
            return None
        repaired[:, axis_idx] = np.interp(times_s, times_s[valid], axis_values[valid])
    return repaired


def resample_values(
    seg_t: np.ndarray,
    seg_values: np.ndarray,
    *,
    target_fs: float,
    cutoff: float,
    filter_order: int,
) -> tuple[np.ndarray, np.ndarray, float, str] | None:
    if len(seg_t) < 2:
        return None

    duration = float(seg_t[-1])
    if duration <= 0.0:
        return None

    target_t = np.arange(0.0, duration, 1.0 / target_fs, dtype=np.float64)
    if len(target_t) == 0:
        return None

    fs = effective_fs(seg_t)
    repaired = interpolate_missing(seg_t, seg_values)
    if repaired is None:
        return None
    filtered = lowpass_if_needed(repaired, fs=fs, cutoff=cutoff, order=filter_order)

    rounded_fs = int(round(fs))
    if fs > target_fs and rounded_fs > 0 and abs(fs - rounded_fs) <= 0.05:
        resampled = resample_poly(filtered, up=int(round(target_fs)), down=rounded_fs, axis=0)
        expected_len = len(target_t)
        if len(resampled) > expected_len:
            resampled = resampled[:expected_len]
        elif 0 < len(resampled) < expected_len:
            pad = np.repeat(resampled[-1:, :], expected_len - len(resampled), axis=0)
            resampled = np.vstack([resampled, pad])
        method = f"poly_{rounded_fs}_to_{int(round(target_fs))}"
    else:
        resampled = np.column_stack(
            [np.interp(target_t, seg_t, filtered[:, axis_idx]) for axis_idx in range(filtered.shape[1])]
        )
        method = "interp"

    return resampled.astype(np.float32, copy=False), target_t, fs, method


def selected_raw_columns() -> list[str]:
    columns = {"timestamp_s", "activity_id"}
    for sensor_cols in SIGNAL_COLUMNS.values():
        columns.update(sensor_cols)
    selected = [name for name in RAW_COLUMN_NAMES if name in columns]
    return selected


def load_raw_subject(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMN_NAMES,
        usecols=selected_raw_columns(),
        na_values=["NaN"],
        engine="c",
    )
    df = df.sort_values("timestamp_s").drop_duplicates("timestamp_s").reset_index(drop=True)
    df["activity_id"] = pd.to_numeric(df["activity_id"], errors="coerce").fillna(0).astype(np.int64)
    return df


def trim_segment_to_common_overlap(seg_df: pd.DataFrame) -> pd.DataFrame | None:
    starts: list[float] = []
    ends: list[float] = []

    for sensor_cols in SIGNAL_COLUMNS.values():
        valid = seg_df[list(sensor_cols)].notna().all(axis=1).to_numpy()
        idx = np.flatnonzero(valid)
        if len(idx) < 2:
            return None
        starts.append(float(seg_df["timestamp_s"].iloc[int(idx[0])]))
        ends.append(float(seg_df["timestamp_s"].iloc[int(idx[-1])]))

    overlap_start = max(starts)
    overlap_end = min(ends)
    if overlap_end <= overlap_start:
        return None

    trimmed = seg_df[(seg_df["timestamp_s"] >= overlap_start) & (seg_df["timestamp_s"] <= overlap_end)].copy()
    if len(trimmed) < 2:
        return None
    return trimmed.reset_index(drop=True)


def iter_signal_keys() -> Iterable[tuple[str, str]]:
    for placement in PLACEMENTS:
        for sensor in SENSORS:
            yield placement, sensor


def build_summary_row(subject_id: str, placement: str, sensor: str, signal: np.ndarray) -> dict:
    row: dict[str, object] = {
        "subject_id": subject_id,
        "sensor": f"{placement}_{sensor}",
    }
    for axis_idx, axis in enumerate(AXES):
        values = signal[:, axis_idx]
        prefix = f"{placement}_{sensor}_{axis}"
        row[f"{prefix}_mean"] = float(np.mean(values))
        row[f"{prefix}_std"] = float(np.std(values))
        row[f"{prefix}_min"] = float(np.min(values))
        row[f"{prefix}_max"] = float(np.max(values))
        row[f"{prefix}_range"] = float(np.ptp(values))
    return row


def rebuild_subject(
    raw_df: pd.DataFrame,
    *,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray, np.ndarray, dict]:
    if len(raw_df) < 2:
        return {}, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), {
            "raw_rows": int(len(raw_df)),
            "raw_span_s": 0.0,
            "n_segments": 0,
            "n_gaps_gt_threshold": 0,
            "raw_label_values": [],
            "segment_fs_values": [],
            "segment_methods": [],
        }

    times_s = raw_df["timestamp_s"].to_numpy(dtype=np.float64)
    labels = raw_df["activity_id"].to_numpy(dtype=np.int64)
    segments = split_at_gaps(times_s, gap_threshold_s)

    signal_segments = {key: [] for key in iter_signal_keys()}
    label_segments: list[np.ndarray] = []
    timestamp_segments: list[np.ndarray] = []
    segment_fs_values: list[float] = []
    segment_methods: list[str] = []

    for segment in segments:
        seg_df = trim_segment_to_common_overlap(raw_df.iloc[segment].reset_index(drop=True))
        if seg_df is None or len(seg_df) < 2:
            continue

        seg_t_abs = seg_df["timestamp_s"].to_numpy(dtype=np.float64)
        seg_t = seg_t_abs - seg_t_abs[0]
        seg_labels = seg_df["activity_id"].to_numpy(dtype=np.int64)

        segment_target_t: np.ndarray | None = None
        segment_fs = 0.0
        segment_method = "empty"
        rebuilt_signals: dict[tuple[str, str], np.ndarray] = {}

        for key in iter_signal_keys():
            sensor_cols = list(SIGNAL_COLUMNS[key])
            seg_values = seg_df[sensor_cols].to_numpy(dtype=np.float64)
            result = resample_values(
                seg_t,
                seg_values,
                target_fs=target_fs,
                cutoff=cutoff,
                filter_order=filter_order,
            )
            if result is None:
                rebuilt_signals = {}
                break
            signal, target_t, fs, method = result
            if len(signal) == 0:
                rebuilt_signals = {}
                break
            rebuilt_signals[key] = signal
            if segment_target_t is None:
                segment_target_t = target_t
                segment_fs = fs
                segment_method = method

        if not rebuilt_signals or segment_target_t is None or len(segment_target_t) == 0:
            continue

        aligned_labels = nearest_labels(seg_t, seg_labels, segment_target_t).astype(np.int64, copy=False)
        for key, signal in rebuilt_signals.items():
            signal_segments[key].append(signal)
        label_segments.append(aligned_labels)
        timestamp_segments.append(segment_target_t)
        segment_fs_values.append(segment_fs)
        segment_methods.append(segment_method)

    if not label_segments:
        return {}, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), {
            "raw_rows": int(len(raw_df)),
            "raw_span_s": float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0,
            "n_segments": 0,
            "n_gaps_gt_threshold": int((np.diff(times_s) > gap_threshold_s).sum()),
            "raw_label_values": sorted(map(int, np.unique(labels).tolist())),
            "segment_fs_values": [],
            "segment_methods": [],
        }

    signals = {key: np.vstack(parts) for key, parts in signal_segments.items()}
    rebuilt_labels = np.concatenate(label_segments)
    rebuilt_timestamps = np.arange(len(rebuilt_labels), dtype=np.float64) / float(target_fs)

    meta = {
        "raw_rows": int(len(raw_df)),
        "raw_span_s": float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0,
        "n_segments": int(len(label_segments)),
        "n_gaps_gt_threshold": int((np.diff(times_s) > gap_threshold_s).sum()),
        "raw_label_values": sorted(map(int, np.unique(labels).tolist())),
        "segment_fs_values": [float(v) for v in segment_fs_values],
        "segment_methods": segment_methods,
    }
    return signals, rebuilt_labels, rebuilt_timestamps, meta


def process_subject(task: tuple) -> tuple[dict, list[dict], Counter, str]:
    (
        raw_path_str,
        output_dir_str,
        target_fs,
        cutoff,
        filter_order,
        gap_threshold_s,
    ) = task

    raw_path = Path(raw_path_str)
    output_dir = Path(output_dir_str)
    subject = raw_path.stem

    raw_df = load_raw_subject(raw_path)
    signals, labels, timestamps, meta = rebuild_subject(
        raw_df,
        target_fs=target_fs,
        cutoff=cutoff,
        filter_order=filter_order,
        gap_threshold_s=gap_threshold_s,
    )

    if len(labels) == 0:
        return (
            {
                "subject_id": subject,
                "status": "skipped_empty",
                "raw_rows": meta["raw_rows"],
                "raw_span_s": meta["raw_span_s"],
                "rebuilt_rows": 0,
                "rebuilt_duration_s": 0.0,
                "n_segments": meta["n_segments"],
                "n_gaps_gt_threshold": meta["n_gaps_gt_threshold"],
                "raw_label_values": ",".join(str(v) for v in meta["raw_label_values"]),
                "rebuilt_label_values": "",
                "segment_fs_values": "",
                "segment_methods": "",
                "label_file": "",
            },
            [],
            Counter(),
            f"skip {subject}: no rebuilt samples",
        )

    duration_for_name = float(timestamps[-1]) if len(timestamps) else 0.0
    label_path = output_dir / f"{DATASET_NAME}_{subject}_labels.parquet"
    pd.DataFrame({"timestamp": timestamps, "label": labels}).to_parquet(label_path, index=False)

    summary_rows: list[dict] = []
    signal_files: dict[str, str] = {}
    for placement, sensor in iter_signal_keys():
        signal = signals[(placement, sensor)]
        signal_name = (
            f"{DATASET_NAME}_{subject}_{placement}_{sensor}_"
            f"{int(round(target_fs))}Hz_{duration_for_name:.2f}s.parquet"
        )
        pd.DataFrame(signal, columns=list(AXES)).to_parquet(output_dir / signal_name, index=False)
        summary_rows.append(build_summary_row(subject, placement, sensor, signal))
        signal_files[f"{placement}_{sensor}"] = signal_name

    summary = {
        "subject_id": subject,
        "status": "ok",
        "raw_rows": meta["raw_rows"],
        "raw_span_s": meta["raw_span_s"],
        "rebuilt_rows": int(len(labels)),
        "rebuilt_duration_s": duration_for_name,
        "n_segments": meta["n_segments"],
        "n_gaps_gt_threshold": meta["n_gaps_gt_threshold"],
        "raw_label_values": ",".join(str(v) for v in meta["raw_label_values"]),
        "rebuilt_label_values": ",".join(str(v) for v in sorted(map(int, np.unique(labels).tolist()))),
        "segment_fs_values": ",".join(f"{v:.4f}" for v in meta["segment_fs_values"]),
        "segment_methods": ",".join(meta["segment_methods"]),
        "label_file": label_path.name,
    }
    summary.update(signal_files)

    return summary, summary_rows, Counter(map(int, labels.tolist())), f"wrote {subject}: {len(labels)} samples"


def build_tasks(
    input_dir: Path,
    output_dir: Path,
    subjects: set[str] | None,
    *,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
) -> list[tuple]:
    tasks = []
    for raw_path in sorted(input_dir.glob("subject*.dat")):
        if subjects is not None and raw_path.stem.lower() not in subjects:
            continue
        tasks.append((str(raw_path), str(output_dir), target_fs, cutoff, filter_order, gap_threshold_s))
    return tasks


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = normalize_subjects(args.subjects)
    tasks = build_tasks(
        args.input_dir,
        args.output_dir,
        subjects,
        target_fs=args.target_fs,
        cutoff=args.cutoff,
        filter_order=args.filter_order,
        gap_threshold_s=args.gap_threshold_s,
    )
    if not tasks:
        raise SystemExit("No PAMAP2 subject .dat files matched the requested input/subject filter.")

    summaries: list[dict] = []
    summary_rows: list[dict] = []
    label_counts: Counter = Counter()
    workers = min(max(1, args.workers), len(tasks))

    if workers == 1:
        for task in tasks:
            summary, rows, counts, message = process_subject(task)
            summaries.append(summary)
            summary_rows.extend(rows)
            label_counts.update(counts)
            print(message)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_subject, task) for task in tasks]
            for future in as_completed(futures):
                summary, rows, counts, message = future.result()
                summaries.append(summary)
                summary_rows.extend(rows)
                label_counts.update(counts)
                print(message)

    summary_df = pd.DataFrame(summaries).sort_values("subject_id").reset_index(drop=True)
    summary_df.to_csv(args.output_dir / args.summary_csv, index=False)

    stats_df = pd.DataFrame(summary_rows)
    if not stats_df.empty:
        stats_df = stats_df.reindex(columns=SUMMARY_COLUMNS).sort_values(["subject_id", "sensor"]).reset_index(drop=True)
    stats_df.to_csv(args.output_dir / f"{DATASET_NAME}_summary_stats.csv", index=False)

    label_df = pd.DataFrame(
        [
            {
                "label": int(label),
                "activity": LABEL_NAMES.get(int(label), f"unknown_{int(label)}"),
                "count": int(count),
            }
            for label, count in sorted(label_counts.items())
        ]
    )
    label_df.to_csv(args.output_dir / args.label_counts_csv, index=False)

    print(f"Wrote summary CSV: {args.output_dir / args.summary_csv}")
    print(f"Wrote stats CSV: {args.output_dir / f'{DATASET_NAME}_summary_stats.csv'}")
    print(f"Wrote label counts CSV: {args.output_dir / args.label_counts_csv}")


if __name__ == "__main__":
    main()
