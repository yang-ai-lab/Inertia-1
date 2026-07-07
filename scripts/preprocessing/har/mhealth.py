#!/usr/bin/env python3
"""Rebuild MHEALTH subject recordings for downstream evaluation.

This follows `scripts/preprocessing/DATASET_AUDIT_PLAYBOOK.md`:
- one output unit per raw subject log (`mHealth_subject*.log`);
- preserve the current `MHEALTH_mHealth_{subject}_...` filename contract and
  the shared-per-subject label parquet convention;
- low-pass filter each signal at 10 Hz before 20 Hz resampling;
- align labels by nearest timestamp, never by Fourier or linear label
  resampling;
- map the raw null class (`0`) to `-1` in rebuilt label parquets;
- write rebuilt artifacts to `processed_rebuild_tmp` (or another temp dir)
  instead of overwriting production data.

Notes specific to MHEALTH:
- raw files contain uniformly sampled 50 Hz logs with no explicit timestamps, so
  the rebuild uses the implicit 50 Hz timeline directly;
- the processed subset includes seven signal streams per subject:
  `chest_acc`, `ecg`, `lankle_{acc,gyro,mag}`, and `rarm_{acc,gyro,mag}`.
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


DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./data/raw/MHEALTHDATASET"))
OUTPUT_PREFIX = "MHEALTH"
RAW_FS = 50.0
AXES = ("x", "y", "z")
SIGNAL_ORDER = (
    "chest_acc",
    "ecg",
    "lankle_acc",
    "lankle_gyro",
    "lankle_mag",
    "rarm_acc",
    "rarm_gyro",
    "rarm_mag",
)

RAW_COLUMN_NAMES = [
    "chest_acc_x",
    "chest_acc_y",
    "chest_acc_z",
    "ecg_x",
    "ecg_y",
    "lankle_acc_x",
    "lankle_acc_y",
    "lankle_acc_z",
    "lankle_gyro_x",
    "lankle_gyro_y",
    "lankle_gyro_z",
    "lankle_mag_x",
    "lankle_mag_y",
    "lankle_mag_z",
    "rarm_acc_x",
    "rarm_acc_y",
    "rarm_acc_z",
    "rarm_gyro_x",
    "rarm_gyro_y",
    "rarm_gyro_z",
    "rarm_mag_x",
    "rarm_mag_y",
    "rarm_mag_z",
    "label",
]

SIGNAL_COLUMNS = {
    "chest_acc": ("chest_acc_x", "chest_acc_y", "chest_acc_z"),
    "ecg": ("ecg_x", "ecg_y"),
    "lankle_acc": ("lankle_acc_x", "lankle_acc_y", "lankle_acc_z"),
    "lankle_gyro": ("lankle_gyro_x", "lankle_gyro_y", "lankle_gyro_z"),
    "lankle_mag": ("lankle_mag_x", "lankle_mag_y", "lankle_mag_z"),
    "rarm_acc": ("rarm_acc_x", "rarm_acc_y", "rarm_acc_z"),
    "rarm_gyro": ("rarm_gyro_x", "rarm_gyro_y", "rarm_gyro_z"),
    "rarm_mag": ("rarm_mag_x", "rarm_mag_y", "rarm_mag_z"),
}

SUMMARY_COLUMNS = ["subject_id", "sensor"]
for sensor_name, sensor_cols in SIGNAL_COLUMNS.items():
    axis_names = AXES[: len(sensor_cols)]
    for axis_name in axis_names:
        prefix = f"{sensor_name}_{axis_name}"
        for stat_name in ("mean", "std", "min", "max", "range", "energy"):
            SUMMARY_COLUMNS.append(f"{prefix}_{stat_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATASET_ROOT / "dataset",
        help="Directory containing raw `mHealth_subject*.log` files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "processed_rebuild_tmp",
        help="Directory to write rebuilt parquet artifacts.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subset of subject IDs to rebuild (for example `subject1 subject10`).",
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
        "--clean-output",
        action="store_true",
        help="Delete existing rebuilt MHEALTH artifacts in the output directory before writing new ones.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="mhealth_rebuild_summary.csv",
        help="Per-subject rebuild summary filename written inside the output directory.",
    )
    parser.add_argument(
        "--label-counts-csv",
        type=str,
        default="mhealth_label_counts.csv",
        help="Global rebuilt label-count CSV filename written inside the output directory.",
    )
    parser.add_argument(
        "--summary-stats-csv",
        type=str,
        default="MHEALTH_summary_stats.csv",
        help="Wide per-subject/per-sensor signal summary CSV filename.",
    )
    parser.add_argument(
        "--label-distribution-csv",
        type=str,
        default="label_distribution_summary.csv",
        help="Per-subject label distribution CSV filename.",
    )
    return parser.parse_args()


def normalize_subjects(subjects: list[str] | None) -> set[str] | None:
    if not subjects:
        return None
    return {subject.strip().lower() for subject in subjects if subject.strip()}


def subject_key(raw_path: Path) -> int:
    stem = raw_path.stem
    suffix = stem.split("subject", 1)[-1]
    return int(suffix)


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


def repair_signal(values: np.ndarray) -> np.ndarray:
    if np.isfinite(values).all():
        return values

    repaired = values.astype(np.float64, copy=True)
    x = np.arange(len(repaired), dtype=np.float64)
    for axis_idx in range(repaired.shape[1]):
        axis_values = repaired[:, axis_idx]
        valid = np.isfinite(axis_values)
        if valid.sum() < 2:
            raise ValueError("Signal axis has fewer than two finite samples.")
        repaired[:, axis_idx] = np.interp(x, x[valid], axis_values[valid])
    return repaired


def build_target_times(n_raw: int, raw_fs: float, target_fs: float) -> np.ndarray:
    if n_raw < 2:
        return np.zeros(0, dtype=np.float64)
    duration = float((n_raw - 1) / raw_fs)
    return np.arange(0.0, duration, 1.0 / target_fs, dtype=np.float64)


def resample_signal(
    values: np.ndarray,
    *,
    raw_fs: float,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    expected_len: int,
) -> np.ndarray:
    repaired = repair_signal(values)
    filtered = lowpass_if_needed(repaired, fs=raw_fs, cutoff=cutoff, order=filter_order)

    rounded_fs = int(round(raw_fs))
    if raw_fs > target_fs and rounded_fs > 0 and abs(raw_fs - rounded_fs) <= 0.05:
        resampled = resample_poly(filtered, up=int(round(target_fs)), down=rounded_fs, axis=0)
        if len(resampled) > expected_len:
            resampled = resampled[:expected_len]
        elif 0 < len(resampled) < expected_len:
            pad = np.repeat(resampled[-1:, :], expected_len - len(resampled), axis=0)
            resampled = np.vstack([resampled, pad])
    else:
        raw_times = np.arange(len(filtered), dtype=np.float64) / raw_fs
        target_times = build_target_times(len(filtered), raw_fs, target_fs)
        resampled = np.column_stack(
            [
                np.interp(target_times, raw_times, filtered[:, axis_idx])
                for axis_idx in range(filtered.shape[1])
            ]
        )

    if len(resampled) != expected_len:
        raise ValueError(f"Expected {expected_len} samples after resampling, got {len(resampled)}.")
    return resampled.astype(np.float32, copy=False)


def subject_token_from_path(raw_path: Path) -> str:
    return raw_path.stem.replace("mHealth_", "")


def label_filename(subject: str) -> str:
    return f"{OUTPUT_PREFIX}_mHealth_{subject}_labels.parquet"


def signal_filename(subject: str, signal_name: str, n_rows: int, target_fs: float) -> str:
    duration_s = n_rows / target_fs
    return f"{OUTPUT_PREFIX}_mHealth_{subject}_{signal_name}_{int(round(target_fs))}Hz_{duration_s:.2f}s.parquet"


def make_stats_row(subject_id: str, sensor: str, values: np.ndarray, axis_names: tuple[str, ...]) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in SUMMARY_COLUMNS}
    row["subject_id"] = subject_id
    row["sensor"] = sensor
    for axis_idx, axis_name in enumerate(axis_names):
        axis_values = values[:, axis_idx].astype(np.float64, copy=False)
        prefix = f"{sensor}_{axis_name}"
        row[f"{prefix}_mean"] = float(np.mean(axis_values))
        row[f"{prefix}_std"] = float(np.std(axis_values))
        row[f"{prefix}_min"] = float(np.min(axis_values))
        row[f"{prefix}_max"] = float(np.max(axis_values))
        row[f"{prefix}_range"] = float(np.max(axis_values) - np.min(axis_values))
        row[f"{prefix}_energy"] = float(np.mean(np.square(axis_values)))
    return row


def load_raw_subject(raw_path: Path) -> pd.DataFrame:
    return pd.read_csv(raw_path, sep=r"\s+", header=None, names=RAW_COLUMN_NAMES)


def process_subject(
    raw_path: Path,
    output_dir: Path,
    *,
    target_fs: float,
    cutoff: float,
    filter_order: int,
) -> dict[str, object]:
    raw_df = load_raw_subject(raw_path)
    subject = subject_token_from_path(raw_path)
    subject_id = raw_path.stem

    raw_times = np.arange(len(raw_df), dtype=np.float64) / RAW_FS
    target_times = build_target_times(len(raw_df), RAW_FS, target_fs)
    if len(target_times) == 0:
        raise ValueError(f"{raw_path.name} is too short to rebuild.")

    raw_labels = (
        pd.to_numeric(raw_df["label"], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
        .to_numpy(copy=True)
    )
    raw_labels[raw_labels == 0] = -1
    aligned_labels = nearest_labels(raw_times, raw_labels, target_times).astype(np.int64, copy=False)

    label_df = pd.DataFrame({"timestamp": target_times.astype(np.float64), "label": aligned_labels})
    label_path = output_dir / label_filename(subject)
    label_df.to_parquet(label_path, index=False)

    stats_rows: list[dict[str, object]] = []
    signal_outputs: list[dict[str, object]] = []
    for signal_name in SIGNAL_ORDER:
        raw_cols = SIGNAL_COLUMNS[signal_name]
        axis_names = AXES[: len(raw_cols)]
        values = raw_df[list(raw_cols)].to_numpy(dtype=np.float64, copy=False)
        rebuilt = resample_signal(
            values,
            raw_fs=RAW_FS,
            target_fs=target_fs,
            cutoff=cutoff,
            filter_order=filter_order,
            expected_len=len(target_times),
        )
        signal_path = output_dir / signal_filename(subject, signal_name, len(rebuilt), target_fs)
        pd.DataFrame(rebuilt, columns=list(axis_names)).to_parquet(signal_path, index=False)
        stats_rows.append(make_stats_row(subject_id, signal_name, rebuilt, axis_names))
        signal_outputs.append(
            {
                "signal": signal_name,
                "path": str(signal_path),
                "rows": int(len(rebuilt)),
            }
        )

    label_counter = Counter(int(v) for v in aligned_labels.tolist())
    return {
        "subject": subject,
        "subject_id": subject_id,
        "raw_rows": int(len(raw_df)),
        "rebuilt_rows": int(len(target_times)),
        "duration_s": float(len(target_times) / target_fs),
        "label_path": str(label_path),
        "signal_outputs": signal_outputs,
        "label_counts": {int(label): int(count) for label, count in sorted(label_counter.items())},
        "stats_rows": stats_rows,
    }


def iter_existing_rebuild_files(output_dir: Path) -> Iterable[Path]:
    patterns = [
        "MHEALTH_mHealth_*.parquet",
        "mhealth_rebuild_summary.csv",
        "mhealth_label_counts.csv",
        "MHEALTH_summary_stats.csv",
        "label_distribution_summary.csv",
    ]
    for pattern in patterns:
        yield from output_dir.glob(pattern)


def clean_output_dir(output_dir: Path) -> None:
    for path in iter_existing_rebuild_files(output_dir):
        path.unlink()


def main() -> None:
    args = parse_args()
    subjects_filter = normalize_subjects(args.subjects)

    raw_files = sorted(args.input_dir.glob("mHealth_subject*.log"), key=subject_key)
    if subjects_filter is not None:
        raw_files = [
            raw_path
            for raw_path in raw_files
            if subject_token_from_path(raw_path).lower() in subjects_filter
        ]

    if not raw_files:
        raise SystemExit("No raw MHEALTH subject logs matched the requested input/filter.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clean_output_dir(args.output_dir)

    worker_count = min(int(args.workers), len(raw_files))
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        future_map = {
            pool.submit(
                process_subject,
                raw_path,
                args.output_dir,
                target_fs=float(args.target_fs),
                cutoff=float(args.cutoff),
                filter_order=int(args.filter_order),
            ): raw_path
            for raw_path in raw_files
        }
        for future in as_completed(future_map):
            raw_path = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Failed to rebuild {raw_path.name}: {exc}") from exc

    results.sort(key=lambda item: int(str(item["subject"]).replace("subject", "")))

    summary_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []
    global_label_counter: Counter[int] = Counter()

    for item in results:
        summary_rows.append(
            {
                "subject": item["subject"],
                "subject_id": item["subject_id"],
                "raw_rows": item["raw_rows"],
                "rebuilt_rows": item["rebuilt_rows"],
                "duration_s": item["duration_s"],
                "label_path": item["label_path"],
                "n_signals": len(item["signal_outputs"]),
            }
        )
        for label, count in dict(item["label_counts"]).items():
            label_rows.append(
                {
                    "subject_id": item["subject_id"],
                    "label": int(label),
                    "count": int(count),
                }
            )
            global_label_counter[int(label)] += int(count)
        stats_rows.extend(item["stats_rows"])

    pd.DataFrame(summary_rows).to_csv(args.output_dir / args.summary_csv, index=False)
    pd.DataFrame(
        [{"label": int(label), "count": int(count)} for label, count in sorted(global_label_counter.items())]
    ).to_csv(args.output_dir / args.label_counts_csv, index=False)
    pd.DataFrame(label_rows).to_csv(args.output_dir / args.label_distribution_csv, index=False)
    pd.DataFrame(stats_rows, columns=SUMMARY_COLUMNS).to_csv(
        args.output_dir / args.summary_stats_csv,
        index=False,
    )

    total_rebuilt_rows = sum(int(item["rebuilt_rows"]) for item in results)
    print(f"Rebuilt {len(results)} subject recordings into {args.output_dir}")
    print(f"Total rebuilt 20 Hz rows: {total_rebuilt_rows}")
    print("Global label counts:")
    for label, count in sorted(global_label_counter.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
