#!/usr/bin/env python3
"""Rebuild Opportunity session recordings for downstream evaluation.

This follows `scripts/preprocessing/DATASET_AUDIT_PLAYBOOK.md`:
- one output unit per raw session `.dat` file (for example `S1-ADL1`);
- preserve the current processed filename contract under the `Opportunity_*`
  prefix and shared-per-session label parquet convention;
- low-pass filter signals at 10 Hz, then resample to a uniform 20 Hz timeline
  derived from raw timestamps;
- align labels by nearest timestamp, never by Fourier or linear label
  interpolation;
- write rebuilt artifacts to `processed_rebuild_tmp` (or another temp dir)
  rather than overwriting production data.

Notes specific to Opportunity:
- raw timestamps are explicit millisecond timestamps at ~30.3 Hz with no large
  inter-recording gaps, so we resample on the timestamp grid directly;
- the processed subset contains only a subset of the raw sensor channels, and
  a few duplicated accelerometer columns in the raw files need explicit
  selection. Those mappings below were validated by diffing against the current
  processed files before the rebuild.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./data/raw/OpportunityUCIDataset"))
OUTPUT_PREFIX = "Opportunity"
AXES = ("x", "y", "z")

# Shared label parquet order in the current processed data.
LABEL_TRACK_COLUMNS: dict[str, int] = {
    "Locomotion": 243,
    "HL_Activity": 244,
    "LL_Left_Arm": 245,
    "LL_Left_Arm_Object": 246,
    "LL_Right_Arm": 247,
    "LL_Right_Arm_Object": 248,
    "ML_Both_Arms": 249,
}

# Zero-based raw column indices selected to reproduce the current processed
# sensor subset. Ambiguous accelerometer placements were chosen by comparing
# candidate raw columns against existing processed files on several sessions:
# - `RKN_acc` matches the `RKN^` columns
# - `LUA_acc` matches the `LUA^` columns
# - `RUA_acc` matches the `RUA_` columns
STREAM_COLUMN_INDICES: dict[tuple[str, str], tuple[int, int, int]] = {
    ("BACK", "acc"): (16, 17, 18),
    ("HIP", "acc"): (4, 5, 6),
    ("LH", "acc"): (13, 14, 15),
    ("LUA", "acc"): (7, 8, 9),
    ("LWR", "acc"): (31, 32, 33),
    ("RH", "acc"): (34, 35, 36),
    ("RKN", "acc"): (1, 2, 3),
    ("RUA", "acc"): (10, 11, 12),
    ("RWR", "acc"): (22, 23, 24),
    ("BACK", "gyro"): (40, 41, 42),
    ("LLA", "gyro"): (92, 93, 94),
    ("LUA", "gyro"): (79, 80, 81),
    ("RLA", "gyro"): (66, 67, 68),
    ("RUA", "gyro"): (53, 54, 55),
    ("BACK", "mag"): (43, 44, 45),
    ("LLA", "mag"): (95, 96, 97),
    ("LUA", "mag"): (82, 83, 84),
    ("RLA", "mag"): (69, 70, 71),
    ("RUA", "mag"): (56, 57, 58),
}

STREAM_ORDER: list[tuple[str, str]] = [
    ("BACK", "acc"),
    ("BACK", "gyro"),
    ("BACK", "mag"),
    ("HIP", "acc"),
    ("LH", "acc"),
    ("LLA", "gyro"),
    ("LLA", "mag"),
    ("LUA", "acc"),
    ("LUA", "gyro"),
    ("LUA", "mag"),
    ("LWR", "acc"),
    ("RH", "acc"),
    ("RKN", "acc"),
    ("RLA", "gyro"),
    ("RLA", "mag"),
    ("RUA", "acc"),
    ("RUA", "gyro"),
    ("RUA", "mag"),
    ("RWR", "acc"),
]

RAW_USECOLS = sorted(
    {0}
    | {col for cols in STREAM_COLUMN_INDICES.values() for col in cols}
    | set(LABEL_TRACK_COLUMNS.values())
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATASET_ROOT / "dataset",
        help="Directory containing raw Opportunity `.dat` session files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "processed_rebuild_tmp",
        help="Directory to write rebuilt parquet artifacts.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional subset of raw session IDs to rebuild (for example S1-ADL1 S2-Drill).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of worker processes for the per-session rebuild.",
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
        default="opportunity_rebuild_summary.csv",
        help="Summary CSV filename written inside the output directory.",
    )
    parser.add_argument(
        "--label-counts-csv",
        type=str,
        default="opportunity_label_counts.csv",
        help="Label distribution CSV filename written inside the output directory.",
    )
    return parser.parse_args()


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


def nearest_assign(raw_times: np.ndarray, raw_values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(raw_times, target_times, side="left")
    idx = np.clip(idx, 0, len(raw_times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(raw_times) - 1)
    use_prev = np.abs(target_times - raw_times[prev_idx]) <= np.abs(raw_times[idx] - target_times)
    nearest_idx = np.where(use_prev, prev_idx, idx)
    return raw_values[nearest_idx]


def sanitize_signal(values: np.ndarray) -> np.ndarray | None:
    # A stream with fewer than two finite samples per axis cannot be faithfully
    # reconstructed at 20 Hz, so we drop it rather than writing synthetic data.
    finite_counts = np.isfinite(values).sum(axis=0)
    if np.any(finite_counts < 2):
        return None

    df = pd.DataFrame(values.astype(np.float64))
    df = df.interpolate(limit_direction="both").ffill().bfill()
    arr = df.to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    return arr / 1000.0


def normalize_label_track(values: np.ndarray) -> np.ndarray:
    numeric = (
        pd.to_numeric(pd.Series(values), errors="coerce")
        .fillna(-1)
        .astype(np.int64)
        .to_numpy(copy=True)
    )
    numeric[numeric == 0] = -1
    return numeric


def load_raw_session(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=RAW_USECOLS)
    df = df.rename(columns={0: "MILLISEC"})
    df["MILLISEC"] = pd.to_numeric(df["MILLISEC"], errors="coerce")
    df = df.dropna(subset=["MILLISEC"]).sort_values("MILLISEC").drop_duplicates("MILLISEC").reset_index(drop=True)
    return df


def rebuild_session(
    raw_df: pd.DataFrame,
    *,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray], dict]:
    if len(raw_df) < 2:
        return {}, {}, {"raw_rows": int(len(raw_df)), "raw_span_s": 0.0, "n_segments": 0}

    times_s = raw_df["MILLISEC"].to_numpy(dtype=np.float64) / 1000.0
    times_s = times_s - times_s[0]
    segments = split_at_gaps(times_s, gap_threshold_s)

    raw_signals = {
        key: raw_df.loc[:, list(cols)].to_numpy(dtype=np.float64)
        for key, cols in STREAM_COLUMN_INDICES.items()
    }
    raw_labels = {
        track: normalize_label_track(raw_df[col].to_numpy())
        for track, col in LABEL_TRACK_COLUMNS.items()
    }

    signal_segments: dict[tuple[str, str], list[np.ndarray]] = {key: [] for key in STREAM_COLUMN_INDICES}
    stream_valid = {key: True for key in STREAM_COLUMN_INDICES}
    label_segments: dict[str, list[np.ndarray]] = {track: [] for track in LABEL_TRACK_COLUMNS}
    segment_fs_values: list[float] = []
    segment_lengths: list[int] = []
    skip_reasons: dict[tuple[str, str], str] = {}

    for segment in segments:
        seg_t_abs = times_s[segment]
        if len(seg_t_abs) < 2:
            continue
        seg_t = seg_t_abs - seg_t_abs[0]
        duration = float(seg_t[-1])
        if duration <= 0.0:
            continue

        target_t = np.arange(0.0, duration, 1.0 / target_fs, dtype=np.float64)
        if len(target_t) == 0:
            continue

        seg_fs = effective_fs(seg_t)
        segment_fs_values.append(seg_fs)
        segment_lengths.append(int(len(target_t)))

        for track in LABEL_TRACK_COLUMNS:
            aligned = nearest_assign(seg_t, raw_labels[track][segment], target_t).astype(np.int64, copy=False)
            label_segments[track].append(aligned)

        for key in STREAM_COLUMN_INDICES:
            if not stream_valid[key]:
                continue

            sanitized = sanitize_signal(raw_signals[key][segment])
            if sanitized is None:
                stream_valid[key] = False
                skip_reasons[key] = "insufficient_finite_samples"
                continue

            filtered = lowpass_if_needed(sanitized, fs=seg_fs, cutoff=cutoff, order=filter_order)
            rebuilt = np.column_stack(
                [np.interp(target_t, seg_t, filtered[:, axis_idx]) for axis_idx in range(filtered.shape[1])]
            ).astype(np.float32, copy=False)
            signal_segments[key].append(rebuilt)

    labels = {
        track: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        for track, chunks in label_segments.items()
    }

    signals = {}
    expected_segments = len([v for v in label_segments["Locomotion"] if len(v) > 0])
    for key, chunks in signal_segments.items():
        if stream_valid[key] and len(chunks) == expected_segments and chunks:
            signals[key] = np.vstack(chunks)
        elif key not in skip_reasons:
            skip_reasons[key] = "missing_one_or_more_segments"

    meta = {
        "raw_rows": int(len(raw_df)),
        "raw_span_s": float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0,
        "n_segments": int(len(segments)),
        "n_gaps_gt_threshold": int((np.diff(times_s) > gap_threshold_s).sum()),
        "segment_fs_values": [float(v) for v in segment_fs_values],
        "segment_lengths": segment_lengths,
        "raw_locomotion_values": sorted(map(int, np.unique(raw_labels["Locomotion"]).tolist())),
        "rebuilt_locomotion_values": sorted(map(int, np.unique(labels["Locomotion"]).tolist()))
        if len(labels["Locomotion"])
        else [],
        "skip_reasons": {f"{placement}_{sensor}": reason for (placement, sensor), reason in skip_reasons.items()},
    }
    return signals, labels, meta


def process_session(task: tuple) -> tuple[dict, Counter, str]:
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
    session = raw_path.stem

    raw_df = load_raw_session(raw_path)
    signals, labels, meta = rebuild_session(
        raw_df,
        target_fs=target_fs,
        cutoff=cutoff,
        filter_order=filter_order,
        gap_threshold_s=gap_threshold_s,
    )

    locomotion = labels.get("Locomotion", np.empty(0, dtype=np.int64))
    if len(locomotion) == 0:
        return (
            {
                "session": session,
                "status": "skipped_empty",
                "raw_rows": meta["raw_rows"],
                "raw_span_s": meta["raw_span_s"],
            },
            Counter(),
            f"skip {session}: no rebuilt labels",
        )

    duration_sec = len(locomotion) / float(target_fs)
    timestamps = np.arange(len(locomotion), dtype=np.float64) / float(target_fs)

    label_df = pd.DataFrame({"timestamp": timestamps})
    for track in LABEL_TRACK_COLUMNS:
        label_df[track] = labels[track]
    label_df["subject_id"] = session
    label_path = output_dir / f"{OUTPUT_PREFIX}_{session}_labels.parquet"
    label_df.to_parquet(label_path, index=False)

    written_streams: list[str] = []
    skipped_streams: list[str] = []
    for placement, sensor in STREAM_ORDER:
        key = (placement, sensor)
        stream_name = f"{placement}_{sensor}"
        signal = signals.get(key)
        if signal is None:
            skipped_streams.append(stream_name)
            continue

        signal_name = (
            f"{OUTPUT_PREFIX}_{session}_{placement}_{sensor}_{int(round(target_fs))}Hz_{duration_sec:.2f}s.parquet"
        )
        pd.DataFrame(signal, columns=list(AXES)).to_parquet(output_dir / signal_name, index=False)
        written_streams.append(stream_name)

    track_counts = Counter()
    for track, values in labels.items():
        track_counts.update({(track, int(label)): int(count) for label, count in Counter(values.tolist()).items()})

    summary = {
        "session": session,
        "status": "ok",
        "raw_rows": meta["raw_rows"],
        "raw_span_s": meta["raw_span_s"],
        "rebuilt_rows": int(len(locomotion)),
        "rebuilt_duration_s": float(duration_sec),
        "n_segments": meta["n_segments"],
        "n_gaps_gt_threshold": meta["n_gaps_gt_threshold"],
        "segment_fs_values": ",".join(f"{v:.4f}" for v in meta["segment_fs_values"]),
        "segment_lengths": ",".join(str(v) for v in meta["segment_lengths"]),
        "raw_locomotion_values": ",".join(str(v) for v in meta["raw_locomotion_values"]),
        "rebuilt_locomotion_values": ",".join(str(v) for v in meta["rebuilt_locomotion_values"]),
        "label_file": label_path.name,
        "written_streams": ",".join(written_streams),
        "skipped_streams": ",".join(skipped_streams),
        "skip_reasons": ";".join(f"{k}:{v}" for k, v in sorted(meta["skip_reasons"].items())),
    }
    return summary, track_counts, f"wrote {session}: {len(locomotion)} labels, {len(written_streams)} signals"


def normalize_sessions(session_ids: list[str] | None) -> set[str] | None:
    if not session_ids:
        return None
    return {session.strip() for session in session_ids if session.strip()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_sessions = normalize_sessions(args.sessions)
    raw_paths = sorted(args.input_dir.glob("S*.dat"))
    if requested_sessions is not None:
        raw_paths = [path for path in raw_paths if path.stem in requested_sessions]

    if not raw_paths:
        raise SystemExit("No raw Opportunity session files matched the requested subset.")

    tasks = [
        (
            str(path),
            str(args.output_dir),
            float(args.target_fs),
            float(args.cutoff),
            int(args.filter_order),
            float(args.gap_threshold_s),
        )
        for path in raw_paths
    ]

    summaries: list[dict] = []
    aggregate_counts: Counter = Counter()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_session, task) for task in tasks]
        for future in as_completed(futures):
            summary, track_counts, message = future.result()
            summaries.append(summary)
            aggregate_counts.update(track_counts)
            print(message, flush=True)

    summaries.sort(key=lambda row: row["session"])
    pd.DataFrame(summaries).to_csv(args.output_dir / args.summary_csv, index=False)

    label_count_rows = [
        {"track": track, "label": label, "count": count}
        for (track, label), count in sorted(aggregate_counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    pd.DataFrame(label_count_rows).to_csv(args.output_dir / args.label_counts_csv, index=False)

    total_rows = sum(int(row.get("rebuilt_rows", 0)) for row in summaries if row.get("status") == "ok")
    total_signals = sum(
        len([name for name in str(row.get("written_streams", "")).split(",") if name])
        for row in summaries
        if row.get("status") == "ok"
    )
    print(f"Completed {len(summaries)} session(s); wrote {total_rows} label rows and {total_signals} signal files.")


if __name__ == "__main__":
    main()
