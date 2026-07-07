#!/usr/bin/env python3
"""Rebuild HAR70Plus subject recordings for downstream evaluation.

This follows `scripts/preprocessing/DATASET_AUDIT_PLAYBOOK.md`:
- one output unit per contiguous post-gap segment within each raw subject CSV;
- keep the current downstream scope: paired `back` + `thigh` accelerometers;
- split recordings at large timestamp gaps before rebuilding;
- low-pass filter signals at 10 Hz, then resample to 20 Hz;
- align labels by nearest timestamp, never by Fourier/linear resampling;
- write rebuilt artifacts to `processed_rebuild_tmp` (or another temp dir),
  using segment-aware filenames so downstream windows cannot cross removed gaps.
"""

from __future__ import annotations

import argparse
import os
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


DATASET_NAME = "HAR70Plus"
AXES = ("x", "y", "z")
PLACEMENTS = ("back", "thigh")
RAW_COLUMNS = [
    "timestamp",
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
    "label",
]


def parse_args() -> argparse.Namespace:
    dataset_root = Path(os.environ.get("DATASET_ROOT", "./data/raw/har70plus"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=dataset_root / "dataset",
        help="Directory containing raw HAR70Plus subject CSV files.",
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
        help="Optional subset of subject IDs to rebuild (e.g. 501 503 518).",
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
        default="har70plus_rebuild_summary.csv",
        help="Summary CSV filename written inside the output directory.",
    )
    parser.add_argument(
        "--label-counts-csv",
        type=str,
        default="har70plus_label_counts.csv",
        help="Label distribution CSV filename written inside the output directory.",
    )
    return parser.parse_args()


def normalize_subjects(subjects: list[str] | None) -> set[str] | None:
    if not subjects:
        return None
    return {subject.strip() for subject in subjects if subject.strip()}


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
    diffs = diffs[diffs > 0]
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


def resample_values(
    seg_t: np.ndarray,
    seg_values: np.ndarray,
    *,
    target_fs: float,
    cutoff: float,
    filter_order: int,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    duration = float(seg_t[-1])
    if duration <= 0.0:
        return np.empty((0, seg_values.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float64), 0.0, "empty"

    target_t = np.arange(0.0, duration + 1e-12, 1.0 / target_fs, dtype=np.float64)
    if len(target_t) == 0:
        return np.empty((0, seg_values.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float64), 0.0, "empty"

    fs = effective_fs(seg_t)
    filtered = lowpass_if_needed(seg_values, fs=fs, cutoff=cutoff, order=filter_order)

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


def load_raw_subject(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=RAW_COLUMNS, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp", "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    labels = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(np.int64)
    df["label"] = labels.astype(str)
    return df


def rebuild_subject(
    raw_df: pd.DataFrame,
    *,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
) -> tuple[list[dict], dict]:
    if len(raw_df) < 2:
        return [], {
            "raw_rows": int(len(raw_df)),
            "raw_span_s": 0.0,
            "n_segments": 0,
            "n_gaps_gt_threshold": 0,
            "raw_label_values": [],
        }

    times_s = (raw_df["timestamp"] - raw_df["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    labels = raw_df["label"].to_numpy(dtype=object)
    values = {
        placement: raw_df[[f"{placement}_{axis}" for axis in AXES]].to_numpy(dtype=np.float64)
        for placement in PLACEMENTS
    }

    rebuilt_segments: list[dict] = []
    segments = split_at_gaps(times_s, gap_threshold_s)

    for segment_idx, segment in enumerate(segments):
        seg_t_abs = times_s[segment]
        if len(seg_t_abs) < 2:
            continue
        seg_t = seg_t_abs - seg_t_abs[0]

        rebuilt_segment: dict[str, np.ndarray] = {}
        target_t: np.ndarray | None = None
        seg_fs = 0.0
        method = "empty"
        for placement in PLACEMENTS:
            segment_signal, placement_target_t, seg_fs, method = resample_values(
                seg_t,
                values[placement][segment],
                target_fs=target_fs,
                cutoff=cutoff,
                filter_order=filter_order,
            )
            if len(segment_signal) == 0:
                rebuilt_segment = {}
                break
            rebuilt_segment[placement] = segment_signal
            if target_t is None:
                target_t = placement_target_t

        if not rebuilt_segment or target_t is None or len(target_t) == 0:
            continue

        aligned_labels = nearest_labels(seg_t, labels[segment], target_t).astype(str, copy=False)
        rebuilt_segments.append({
            "segment_idx": int(segment_idx),
            "segment_token": f"seg{segment_idx:03d}",
            "raw_start_s": float(seg_t_abs[0]),
            "raw_end_s": float(seg_t_abs[-1]),
            "raw_duration_s": float(seg_t[-1]),
            "raw_rows": int(len(seg_t_abs)),
            "rebuilt_rows": int(len(target_t)),
            "rebuilt_duration_s": float(len(target_t) / target_fs),
            "segment_fs": float(seg_fs),
            "segment_method": method,
            "signals": rebuilt_segment,
            "labels": aligned_labels,
        })

    if not rebuilt_segments:
        return [], {
            "raw_rows": int(len(raw_df)),
            "raw_span_s": float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0,
            "n_segments": 0,
            "n_gaps_gt_threshold": int((np.diff(times_s) > gap_threshold_s).sum()),
            "raw_label_values": sorted(map(str, pd.unique(labels).tolist())),
        }

    meta = {
        "raw_rows": int(len(raw_df)),
        "raw_span_s": float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0,
        "n_segments": int(len(rebuilt_segments)),
        "n_gaps_gt_threshold": int((np.diff(times_s) > gap_threshold_s).sum()),
        "raw_label_values": sorted(map(str, pd.unique(labels).tolist())),
    }
    return rebuilt_segments, meta


def process_subject(task: tuple) -> tuple[list[dict], Counter, str]:
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
    segments, meta = rebuild_subject(
        raw_df,
        target_fs=target_fs,
        cutoff=cutoff,
        filter_order=filter_order,
        gap_threshold_s=gap_threshold_s,
    )

    if not segments:
        return (
            [{
                "subject": subject,
                "segment_token": "",
                "status": "skipped_empty",
                "raw_rows_subject": meta["raw_rows"],
                "raw_span_s_subject": meta["raw_span_s"],
                "subject_segment_count": 0,
                "subject_gap_count": meta["n_gaps_gt_threshold"],
            }],
            Counter(),
            f"skip {subject}: no rebuilt samples",
        )

    summaries: list[dict] = []
    label_counts: Counter = Counter()
    for segment in segments:
        seg_token = segment["segment_token"]
        labels = segment["labels"]
        timestamps = np.arange(len(labels), dtype=np.float64) / float(target_fs)

        label_path = output_dir / f"{DATASET_NAME}_{subject}_{seg_token}_labels.parquet"
        pd.DataFrame({"subject_id": subject, "timestamp_sec": timestamps, "label": labels}).to_parquet(
            label_path, index=False
        )

        signal_paths: dict[str, str] = {}
        for placement in PLACEMENTS:
            signal_name = (
                f"{DATASET_NAME}_{subject}_{seg_token}_{placement}_acc_"
                f"{int(round(target_fs))}Hz_{segment['rebuilt_duration_s']:.2f}s.parquet"
            )
            pd.DataFrame(segment["signals"][placement], columns=list(AXES)).to_parquet(output_dir / signal_name, index=False)
            signal_paths[placement] = signal_name

        summaries.append({
            "subject": subject,
            "segment_token": seg_token,
            "status": "ok",
            "raw_rows_subject": meta["raw_rows"],
            "raw_span_s_subject": meta["raw_span_s"],
            "subject_segment_count": meta["n_segments"],
            "subject_gap_count": meta["n_gaps_gt_threshold"],
            "raw_label_values_subject": ",".join(meta["raw_label_values"]),
            "raw_segment_rows": segment["raw_rows"],
            "raw_start_s": segment["raw_start_s"],
            "raw_end_s": segment["raw_end_s"],
            "raw_duration_s": segment["raw_duration_s"],
            "raw_effective_fs": segment["segment_fs"],
            "segment_method": segment["segment_method"],
            "rebuilt_rows": segment["rebuilt_rows"],
            "rebuilt_duration_s": segment["rebuilt_duration_s"],
            "rebuilt_label_values": ",".join(sorted(pd.unique(labels).tolist())),
            "label_file": label_path.name,
            "back_file": signal_paths["back"],
            "thigh_file": signal_paths["thigh"],
        })
        label_counts.update(labels.tolist())

    return summaries, label_counts, f"wrote {subject}: {len(segments)} segment files"


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
    for raw_path in sorted(input_dir.glob("*.csv")):
        subject = raw_path.stem
        if subjects is not None and subject not in subjects:
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
        raise SystemExit("No HAR70Plus subject CSVs matched the requested input/subject filter.")

    summaries: list[dict] = []
    label_counts: Counter = Counter()
    workers = min(max(1, args.workers), len(tasks))

    if workers == 1:
        for task in tasks:
            summary_rows, counts, message = process_subject(task)
            summaries.extend(summary_rows)
            label_counts.update(counts)
            print(message)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_subject, task) for task in tasks]
            for future in as_completed(futures):
                summary_rows, counts, message = future.result()
                summaries.extend(summary_rows)
                label_counts.update(counts)
                print(message)

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["subject", "segment_token"]).reset_index(drop=True)
    summary_path = args.output_dir / args.summary_csv
    summary_df.to_csv(summary_path, index=False)

    label_df = pd.DataFrame(
        [{"label": str(label), "count": int(count)} for label, count in sorted(label_counts.items(), key=lambda x: str(x[0]))]
    )
    label_counts_path = args.output_dir / args.label_counts_csv
    label_df.to_csv(label_counts_path, index=False)

    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote label counts CSV: {label_counts_path}")


if __name__ == "__main__":
    main()
