#!/usr/bin/env python3
"""Rebuild WISDM watch recordings for downstream evaluation.

This follows `scripts/preprocessing/DATASET_AUDIT_PLAYBOOK.md` for the
explicitly approved watch-only WISDM scope:
- keep watch accelerometer + gyroscope only;
- exclude all phone files;
- rebuild one output per raw subject/sensor recording;
- emit 20 Hz watch parquets with sibling label files;
- align labels by nearest timestamp rather than resampling categories.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


AXES = ["x", "y", "z"]
WATCH_SENSORS = ("accel", "gyro")


def resolve_watch_root(input_dir: str | Path) -> Path:
    root = Path(input_dir)
    if all((root / sensor).is_dir() for sensor in WATCH_SENSORS):
        return root

    watch_root = root / "watch"
    if all((watch_root / sensor).is_dir() for sensor in WATCH_SENSORS):
        return watch_root

    raise FileNotFoundError(
        f"Could not find watch accel/gyro folders under {root}. "
        "Pass either the raw root containing watch/ or the watch directory itself."
    )


def load_raw_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=["subject", "activity", "timestamp", "x", "y", "z"],
        sep=",",
        engine="python",
        comment="#",
    ).dropna(how="all")

    for col in AXES:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(";", "", regex=False)
            .str.strip()
            .replace("", np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["subject"] = pd.to_numeric(df["subject"], errors="coerce")
    df["activity"] = df["activity"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["subject", "timestamp", *AXES]).copy()
    df["subject"] = df["subject"].astype(int)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def split_at_gaps(times_s: np.ndarray, gap_threshold_s: float) -> list[slice]:
    if len(times_s) == 0:
        return []
    gaps = np.flatnonzero(np.diff(times_s) > gap_threshold_s) + 1
    starts = np.r_[0, gaps]
    ends = np.r_[gaps, len(times_s)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def effective_fs(times_s: np.ndarray) -> float:
    duration = float(times_s[-1] - times_s[0])
    if len(times_s) < 2 or duration <= 0:
        return 0.0
    return float((len(times_s) - 1) / duration)


def lowpass_if_needed(values: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    if fs <= 20.0 or len(values) <= (order * 3 + 1):
        return values

    nyq = 0.5 * fs
    cutoff_eff = min(cutoff, np.nextafter(nyq, 0.0))
    if cutoff_eff <= 0:
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
    nearest = np.where(use_prev, prev_idx, idx)
    return raw_labels[nearest]


def resample_recording(
    df: pd.DataFrame,
    *,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(df) < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=object), {"raw_fs": 0.0, "segments": 0}

    times_s = df["timestamp"].to_numpy(dtype=np.float64) / 1e9
    values = df[AXES].to_numpy(dtype=np.float64)
    labels = df["activity"].fillna("").astype(str).to_numpy(dtype=object)

    signal_segments: list[np.ndarray] = []
    label_segments: list[np.ndarray] = []
    raw_fs_values: list[float] = []

    for segment in split_at_gaps(times_s, gap_threshold_s):
        seg_t = times_s[segment]
        seg_values = values[segment]
        seg_labels = labels[segment]
        if len(seg_t) < 2:
            continue

        seg_t = seg_t - seg_t[0]
        duration = float(seg_t[-1])
        if duration <= 0:
            continue

        target_t = np.arange(0.0, duration, 1.0 / target_fs, dtype=np.float64)
        if len(target_t) == 0:
            continue

        fs = effective_fs(seg_t)
        raw_fs_values.append(fs)
        filtered = lowpass_if_needed(seg_values, fs=fs, cutoff=cutoff, order=filter_order)
        resampled = np.column_stack([
            np.interp(target_t, seg_t, filtered[:, axis_idx])
            for axis_idx in range(filtered.shape[1])
        ])
        aligned_labels = nearest_labels(seg_t, seg_labels, target_t)

        signal_segments.append(resampled.astype(np.float32, copy=False))
        label_segments.append(aligned_labels.astype(object, copy=False))

    if not signal_segments:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=object), {"raw_fs": 0.0, "segments": 0}

    meta = {
        "raw_fs": float(np.median(raw_fs_values)) if raw_fs_values else 0.0,
        "segments": len(signal_segments),
    }
    return np.vstack(signal_segments), np.concatenate(label_segments), meta


def compute_summary_stats(
    values: np.ndarray,
    *,
    subject: str,
    sensor: str,
    duration_sec: float,
    raw_fs: float,
    segments: int,
) -> dict:
    stats = {
        "subject_id": subject,
        "placement": "watch",
        "sensor": sensor,
        "duration_sec": float(duration_sec),
        "raw_effective_fs": float(raw_fs),
        "segments": int(segments),
        "n_samples": int(len(values)),
    }
    for idx, axis in enumerate(AXES):
        v = values[:, idx]
        stats.update({
            f"watch_{sensor}_{axis}_mean": float(np.mean(v)),
            f"watch_{sensor}_{axis}_std": float(np.std(v)),
            f"watch_{sensor}_{axis}_min": float(np.min(v)),
            f"watch_{sensor}_{axis}_max": float(np.max(v)),
            f"watch_{sensor}_{axis}_range": float(np.ptp(v)),
        })
    return stats


def subject_from_path(path: Path) -> str:
    return path.stem.split("_")[1]


def build_tasks(
    watch_root: Path,
    output_dir: Path,
    dataset_name: str,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
    min_samples: int,
) -> list[tuple]:
    accel_files = {subject_from_path(path): path for path in sorted((watch_root / "accel").glob("*.txt"))}
    gyro_files = {subject_from_path(path): path for path in sorted((watch_root / "gyro").glob("*.txt"))}

    shared_subjects = sorted(set(accel_files) & set(gyro_files))
    for subject in sorted(set(gyro_files) - set(accel_files)):
        print(f"WARNING: skipping S{subject}: missing watch accel")
    for subject in sorted(set(accel_files) - set(gyro_files)):
        print(f"WARNING: skipping S{subject}: missing watch gyro")

    tasks = []
    for subject in shared_subjects:
        tasks.append(
            (
                subject,
                str(accel_files[subject]),
                str(gyro_files[subject]),
                str(output_dir),
                dataset_name,
                target_fs,
                cutoff,
                filter_order,
                gap_threshold_s,
                min_samples,
            )
        )
    return tasks


def process_subject_pair(task: tuple) -> tuple[list[dict], list[pd.DataFrame], str]:
    (
        subject,
        accel_path,
        gyro_path,
        output_dir,
        dataset_name,
        target_fs,
        cutoff,
        filter_order,
        gap_threshold_s,
        min_samples,
    ) = task

    accel_df = load_raw_file(Path(accel_path))
    gyro_df = load_raw_file(Path(gyro_path))
    if accel_df.empty or gyro_df.empty:
        return [], [], f"skip S{subject}: empty watch accel/gyro file"

    overlap_start = max(accel_df["timestamp"].min(), gyro_df["timestamp"].min())
    overlap_end = min(accel_df["timestamp"].max(), gyro_df["timestamp"].max())
    if overlap_end <= overlap_start:
        return [], [], f"skip S{subject}: no watch accel/gyro overlap"

    accel_df = accel_df[
        (accel_df["timestamp"] >= overlap_start) & (accel_df["timestamp"] <= overlap_end)
    ].reset_index(drop=True)
    gyro_df = gyro_df[
        (gyro_df["timestamp"] >= overlap_start) & (gyro_df["timestamp"] <= overlap_end)
    ].reset_index(drop=True)

    accel_signal, accel_labels, accel_meta = resample_recording(
        accel_df,
        target_fs=target_fs,
        cutoff=cutoff,
        filter_order=filter_order,
        gap_threshold_s=gap_threshold_s,
    )
    gyro_signal, gyro_labels, gyro_meta = resample_recording(
        gyro_df,
        target_fs=target_fs,
        cutoff=cutoff,
        filter_order=filter_order,
        gap_threshold_s=gap_threshold_s,
    )

    min_len = min(len(accel_signal), len(gyro_signal), len(accel_labels), len(gyro_labels))
    if min_len < min_samples:
        return [], [], f"skip S{subject}: only {min_len} aligned watch samples"

    duration_sec = min_len / target_fs
    timestamp = np.arange(min_len, dtype=np.float64) / target_fs
    output_path = Path(output_dir)

    outputs = {
        "accel": {
            "signal": accel_signal[:min_len],
            "labels": accel_labels[:min_len],
            "meta": accel_meta,
        },
        "gyro": {
            "signal": gyro_signal[:min_len],
            "labels": gyro_labels[:min_len],
            "meta": gyro_meta,
        },
    }

    summaries: list[dict] = []
    label_frames: list[pd.DataFrame] = []
    for sensor, payload in outputs.items():
        base = f"{dataset_name}_S{subject}_watch_{sensor}_{int(target_fs)}Hz_{duration_sec:.2f}s"
        signal_path = output_path / f"{base}.parquet"
        label_path = output_path / f"{base}_labels.parquet"

        pd.DataFrame(payload["signal"], columns=AXES).to_parquet(signal_path, index=False)
        label_df = pd.DataFrame({
            "subject_id": subject,
            "timestamp": timestamp,
            "activity": payload["labels"],
            "device": "watch",
            "sensor": sensor,
        })
        label_df.to_parquet(label_path, index=False)

        summary = compute_summary_stats(
            payload["signal"],
            subject=subject,
            sensor=sensor,
            duration_sec=duration_sec,
            raw_fs=payload["meta"]["raw_fs"],
            segments=payload["meta"]["segments"],
        )
        summary.update({
            "signal_file": signal_path.name,
            "label_file": label_path.name,
        })
        summaries.append(summary)
        label_frames.append(label_df)

    return summaries, label_frames, f"wrote watch files for S{subject} ({min_len} samples per sensor)"


def plot_label_summary(labels_df: pd.DataFrame, output_dir: Path) -> None:
    counts = labels_df["activity"].value_counts()
    plt.figure(figsize=(9, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Activity")
    plt.ylabel("Count")
    plt.title("WISDM Watch Label Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png")
    plt.close()


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    dataset_name: str = "WISDM",
    target_fs: float = 20.0,
    cutoff: float = 10.0,
    filter_order: int = 4,
    gap_threshold_s: float = 1.0,
    min_samples: int = 200,
    num_workers: int = 24,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    watch_root = resolve_watch_root(input_dir)
    tasks = build_tasks(
        watch_root=watch_root,
        output_dir=output_path,
        dataset_name=dataset_name,
        target_fs=target_fs,
        cutoff=cutoff,
        filter_order=filter_order,
        gap_threshold_s=gap_threshold_s,
        min_samples=min_samples,
    )
    print(f"Processing {len(tasks)} watch subject pairs with {num_workers} workers")

    all_summaries: list[dict] = []
    all_labels: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_subject_pair, task) for task in tasks]
        for future in as_completed(futures):
            summaries, label_frames, message = future.result()
            print(message)
            all_summaries.extend(summaries)
            all_labels.extend(label_frames)

    if all_summaries:
        pd.DataFrame(all_summaries).sort_values(["subject_id", "sensor"]).to_csv(
            output_path / f"{dataset_name}_summary_stats.csv",
            index=False,
        )

    if all_labels:
        labels_df = pd.concat(all_labels, ignore_index=True)
        labels_df.to_csv(output_path / f"{dataset_name}_all_labels.csv", index=False)
        plot_label_summary(labels_df, output_path)

    print(f"Completed WISDM watch preprocessing: {len(all_summaries)} signal files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess WISDM watch accel/gyro data into 20 Hz parquets.")
    parser.add_argument("--input_dir", required=True, help="Raw root containing watch/ or the watch directory itself.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="WISDM")
    parser.add_argument("--target_fs", type=float, default=20.0)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--filter_order", type=int, default=4)
    parser.add_argument("--gap_threshold_s", type=float, default=1.0)
    parser.add_argument("--min_samples", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=24)
    args = parser.parse_args()

    preprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        target_fs=args.target_fs,
        cutoff=args.cutoff,
        filter_order=args.filter_order,
        gap_threshold_s=args.gap_threshold_s,
        min_samples=args.min_samples,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()