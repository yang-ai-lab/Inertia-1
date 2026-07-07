#!/usr/bin/env python3
"""Rebuild the audited Daphnet FoG processed subset for downstream evaluation.

This follows `scripts/preprocessing/DATASET_AUDIT_PLAYBOOK_GAIT.md` for
`daphnet_fog`:
- raw recordings live as one whitespace-delimited TXT file per `(subject, run)`;
- columns are timestamp milliseconds, ankle/thigh/trunk 3-axis acceleration in
  mg, and raw annotations;
- raw annotations are binarized for gait detection as `0 -> -1` (outside
  experiment), `1 -> 0` (experiment/no freeze), and `2 -> 1` (freeze);
- signals are rebuilt at 20 Hz from the raw 64 Hz timeline with a 4th-order
  10 Hz Butterworth low-pass filter followed by `resample_poly`;
- labels are rebuilt by nearest-neighbor assignment onto the rebuilt 20 Hz
  timeline;
- the existing filename contract is preserved: one shared label parquet per
  `SxxRyy` recording and one signal parquet for each of ankle/thigh/trunk.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


DATASET_NAME = "DaphnetFOG"
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./data/raw/daphnet_fog"))
RAW_DIR = DATASET_ROOT / "dataset"
OUTPUT_DIR = DATASET_ROOT / "processed_rebuild_tmp"

TARGET_FS = 20.0
RAW_FS = 64.0
FILTER_CUTOFF_HZ = 10.0
FILTER_ORDER = 4
GAP_THRESHOLD_S = 1.0
AXES = ("x", "y", "z")
PLACEMENT_COLUMNS = {
    "ankle": (1, 2, 3),
    "thigh": (4, 5, 6),
    "trunk": (7, 8, 9),
}
RAW_COLUMNS = [
    "timestamp_ms",
    "ankle_x",
    "ankle_y",
    "ankle_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
    "trunk_x",
    "trunk_y",
    "trunk_z",
    "annotation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--recordings", nargs="*", default=None, help="Optional SxxRyy recording IDs to rebuild.")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--target-fs", type=float, default=TARGET_FS)
    parser.add_argument("--raw-fs", type=float, default=RAW_FS)
    parser.add_argument("--cutoff", type=float, default=FILTER_CUTOFF_HZ)
    parser.add_argument("--filter-order", type=int, default=FILTER_ORDER)
    parser.add_argument("--gap-threshold-s", type=float, default=GAP_THRESHOLD_S)
    parser.add_argument("--summary-csv", type=str, default="daphnet_fog_rebuild_summary.csv")
    parser.add_argument("--label-counts-csv", type=str, default="daphnet_fog_label_counts.csv")
    return parser.parse_args()


def normalize_recordings(recordings: list[str] | None) -> set[str] | None:
    if not recordings:
        return None
    return {recording.strip().upper() for recording in recordings if recording.strip()}


def remove_existing_contents(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def split_at_gaps(times_s: np.ndarray, gap_threshold_s: float) -> list[slice]:
    if len(times_s) == 0:
        return []
    gaps = np.flatnonzero(np.diff(times_s) > gap_threshold_s) + 1
    starts = np.r_[0, gaps]
    ends = np.r_[gaps, len(times_s)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def lowpass_if_needed(values: np.ndarray, fs: float, target_fs: float, cutoff: float, order: int) -> np.ndarray:
    if fs <= target_fs or len(values) <= (order * 3 + 1):
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


def nearest_labels(raw_times_s: np.ndarray, raw_labels: np.ndarray, target_times_s: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(raw_times_s, target_times_s, side="left")
    idx = np.clip(idx, 0, len(raw_times_s) - 1)
    prev_idx = np.clip(idx - 1, 0, len(raw_times_s) - 1)
    use_prev = np.abs(target_times_s - raw_times_s[prev_idx]) <= np.abs(raw_times_s[idx] - target_times_s)
    nearest_idx = np.where(use_prev, prev_idx, idx)
    return raw_labels[nearest_idx]


def binarize_annotations(raw_annotations: np.ndarray) -> np.ndarray:
    raw_annotations = raw_annotations.astype(np.int64, copy=False)
    invalid = ~np.isin(raw_annotations, [0, 1, 2])
    if invalid.any():
        bad = sorted(map(int, np.unique(raw_annotations[invalid]).tolist()))
        raise ValueError(f"Unexpected Daphnet annotation values: {bad}")
    return np.where(raw_annotations == 0, -1, np.where(raw_annotations == 2, 1, 0)).astype(np.int64)


def load_raw_recording(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
    df = df.dropna(subset=RAW_COLUMNS).sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    df["annotation"] = pd.to_numeric(df["annotation"], errors="raise").astype(np.int64)
    return df.reset_index(drop=True)


def resample_segment(
    seg_values: np.ndarray,
    target_len: int,
    *,
    raw_fs: float,
    target_fs: float,
    cutoff: float,
    filter_order: int,
) -> np.ndarray:
    filtered = lowpass_if_needed(seg_values, fs=raw_fs, target_fs=target_fs, cutoff=cutoff, order=filter_order)
    resampled = resample_poly(
        filtered,
        up=int(round(target_fs)),
        down=int(round(raw_fs)),
        axis=0,
    )
    if len(resampled) > target_len:
        resampled = resampled[:target_len]
    elif 0 < len(resampled) < target_len:
        pad = np.repeat(resampled[-1:, :], target_len - len(resampled), axis=0)
        resampled = np.vstack([resampled, pad])
    return resampled.astype(np.float32, copy=False)


def rebuild_recording(
    raw_df: pd.DataFrame,
    *,
    raw_fs: float,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    if len(raw_df) < 2:
        empty_signal = np.empty((0, 3), dtype=np.float32)
        return {placement: empty_signal for placement in PLACEMENT_COLUMNS}, np.empty(0, dtype=np.int64), np.empty(0), {
            "raw_rows": int(len(raw_df)),
            "raw_span_s": 0.0,
            "n_segments": 0,
            "n_gaps_gt_threshold": 0,
            "raw_label_values": [],
        }

    raw_times_abs_s = raw_df["timestamp_ms"].to_numpy(dtype=np.float64) / 1000.0
    raw_times_rel_s = raw_times_abs_s - raw_times_abs_s[0]
    raw_annotations = raw_df["annotation"].to_numpy(dtype=np.int64)
    binary_labels = binarize_annotations(raw_annotations)

    signal_segments = {placement: [] for placement in PLACEMENT_COLUMNS}
    label_segments: list[np.ndarray] = []
    timestamp_segments: list[np.ndarray] = []

    for segment in split_at_gaps(raw_times_rel_s, gap_threshold_s):
        seg_times_abs = raw_times_abs_s[segment]
        seg_times_rel = raw_times_rel_s[segment]
        if len(seg_times_rel) < 2:
            continue

        segment_start_abs = float(seg_times_abs[0])
        segment_t = seg_times_rel - seg_times_rel[0]
        target_t_rel = np.arange(0.0, segment_t[-1], 1.0 / target_fs, dtype=np.float64)
        if len(target_t_rel) == 0:
            continue
        target_t_abs = segment_start_abs + target_t_rel
        target_len = len(target_t_rel)

        for placement, cols in PLACEMENT_COLUMNS.items():
            # Daphnet stores acceleration in mg; downstream processed files use g.
            seg_values = raw_df.iloc[segment, list(cols)].to_numpy(dtype=np.float64) / 1000.0
            signal_segments[placement].append(
                resample_segment(
                    seg_values,
                    target_len,
                    raw_fs=raw_fs,
                    target_fs=target_fs,
                    cutoff=cutoff,
                    filter_order=filter_order,
                )
            )

        label_segments.append(nearest_labels(seg_times_abs, binary_labels[segment], target_t_abs))
        timestamp_segments.append(target_t_abs)

    if not label_segments:
        empty_signal = np.empty((0, 3), dtype=np.float32)
        return {placement: empty_signal for placement in PLACEMENT_COLUMNS}, np.empty(0, dtype=np.int64), np.empty(0), {
            "raw_rows": int(len(raw_df)),
            "raw_span_s": float(raw_times_rel_s[-1]),
            "n_segments": 0,
            "n_gaps_gt_threshold": int((np.diff(raw_times_rel_s) > gap_threshold_s).sum()),
            "raw_label_values": sorted(map(int, np.unique(raw_annotations).tolist())),
        }

    signals = {placement: np.vstack(parts) for placement, parts in signal_segments.items()}
    labels = np.concatenate(label_segments).astype(np.int64, copy=False)
    timestamps = np.concatenate(timestamp_segments).astype(np.float64, copy=False)

    meta = {
        "raw_rows": int(len(raw_df)),
        "raw_span_s": float(raw_times_rel_s[-1]),
        "n_segments": int(len(label_segments)),
        "n_gaps_gt_threshold": int((np.diff(raw_times_rel_s) > gap_threshold_s).sum()),
        "raw_label_values": sorted(map(int, np.unique(raw_annotations).tolist())),
    }
    return signals, labels, timestamps, meta


def process_recording(task: tuple) -> tuple[dict, Counter, str]:
    (
        raw_path_str,
        output_dir_str,
        raw_fs,
        target_fs,
        cutoff,
        filter_order,
        gap_threshold_s,
    ) = task
    raw_path = Path(raw_path_str)
    output_dir = Path(output_dir_str)
    recording = raw_path.stem.upper()

    raw_df = load_raw_recording(raw_path)
    signals, labels, timestamps, meta = rebuild_recording(
        raw_df,
        raw_fs=raw_fs,
        target_fs=target_fs,
        cutoff=cutoff,
        filter_order=filter_order,
        gap_threshold_s=gap_threshold_s,
    )

    if len(labels) == 0:
        return (
            {
                "recording": recording,
                "status": "skipped_empty",
                "raw_rows": meta["raw_rows"],
                "raw_span_s": meta["raw_span_s"],
            },
            Counter(),
            f"skip {recording}: no rebuilt samples",
        )

    duration_sec = len(labels) / float(target_fs)
    label_name = f"{DATASET_NAME}_{recording}_labels.parquet"
    pd.DataFrame({"timestamp_s": timestamps, "label": labels}).to_parquet(output_dir / label_name, index=False)

    signal_paths: dict[str, str] = {}
    for placement in PLACEMENT_COLUMNS:
        signal_name = f"{DATASET_NAME}_{recording}_{placement}_acc_{int(round(target_fs))}Hz_{duration_sec:.2f}s.parquet"
        pd.DataFrame(signals[placement], columns=list(AXES)).to_parquet(output_dir / signal_name, index=False)
        signal_paths[placement] = signal_name

    summary = {
        "recording": recording,
        "status": "ok",
        "raw_rows": meta["raw_rows"],
        "raw_span_s": meta["raw_span_s"],
        "rebuilt_rows": int(len(labels)),
        "rebuilt_duration_s": float(duration_sec),
        "n_segments": meta["n_segments"],
        "n_gaps_gt_threshold": meta["n_gaps_gt_threshold"],
        "raw_label_values": ",".join(str(v) for v in meta["raw_label_values"]),
        "rebuilt_label_values": ",".join(str(v) for v in sorted(map(int, np.unique(labels).tolist()))),
        "label_file": label_name,
        "ankle_file": signal_paths["ankle"],
        "thigh_file": signal_paths["thigh"],
        "trunk_file": signal_paths["trunk"],
    }
    return summary, Counter(map(int, labels.tolist())), f"wrote {recording}: {len(labels)} samples"


def build_tasks(
    raw_dir: Path,
    output_dir: Path,
    recordings: set[str] | None,
    *,
    raw_fs: float,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
) -> list[tuple]:
    tasks = []
    for raw_path in sorted(raw_dir.glob("S*.txt")):
        recording = raw_path.stem.upper()
        if recordings is not None and recording not in recordings:
            continue
        tasks.append((str(raw_path), str(output_dir), raw_fs, target_fs, cutoff, filter_order, gap_threshold_s))
    return tasks


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        remove_existing_contents(args.output_dir)

    recordings = normalize_recordings(args.recordings)
    tasks = build_tasks(
        args.raw_dir,
        args.output_dir,
        recordings,
        raw_fs=float(args.raw_fs),
        target_fs=float(args.target_fs),
        cutoff=float(args.cutoff),
        filter_order=int(args.filter_order),
        gap_threshold_s=float(args.gap_threshold_s),
    )
    if not tasks:
        raise SystemExit("No Daphnet recordings matched the requested selection.")

    summaries: list[dict] = []
    total_counts: Counter = Counter()

    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(process_recording, task) for task in tasks]
        for future in as_completed(futures):
            summary, counts, message = future.result()
            summaries.append(summary)
            total_counts.update(counts)
            print(message)

    summaries = sorted(summaries, key=lambda item: item["recording"])
    pd.DataFrame(summaries).to_csv(args.output_dir / args.summary_csv, index=False)
    pd.DataFrame(
        [{"label": int(label), "count": int(count)} for label, count in sorted(total_counts.items())]
    ).to_csv(args.output_dir / args.label_counts_csv, index=False)

    print(f"Wrote {len(summaries)} recording summaries to {args.output_dir / args.summary_csv}")
    print("Label counts:", dict(sorted(total_counts.items())))


if __name__ == "__main__":
    main()
