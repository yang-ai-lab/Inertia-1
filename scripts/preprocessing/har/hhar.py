#!/usr/bin/env python3
"""Preprocess HHAR watch IMU recordings for downstream benchmarking.

This rebuild follows scripts/preprocessing/DATASET_AUDIT_PLAYBOOK.md:
- watch-only, matching the current downstream scope;
- one output per raw (User, Device, Sensor) recording, never merged by user;
- timestamp-based 20 Hz signal output;
- nearest-neighbor label alignment with fixed HHAR label encoding.
"""

import argparse
import os
import re
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
LABEL_MAP = {
    "stand": 0,
    "sit": 1,
    "stairsdown": 2,
    "stairsup": 3,
    "bike": 4,
    "walk": 5,
}


def sanitize_token(value) -> str:
    """Keep filename tokens parseable by the existing standard loader."""
    return re.sub(r"[^A-Za-z0-9.-]+", "", str(value))


def encode_labels(labels: pd.Series) -> np.ndarray:
    normalized = labels.fillna("null").astype(str).str.strip().str.lower()
    return normalized.map(LABEL_MAP).fillna(-1).astype(np.int64).to_numpy()


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
    # scipy.signal.butter requires Wn < 1. Keep the project cutoff at 10 Hz,
    # but avoid the exact-Nyquist no-op/ValueError boundary.
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
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    df = df.sort_values("Creation_Time").drop_duplicates("Creation_Time")
    if len(df) < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int64)

    times_s = df["Creation_Time"].to_numpy(dtype=np.float64) / 1e9
    values = df[AXES].to_numpy(dtype=np.float64)
    labels = encode_labels(df["gt"])

    signal_segments: list[np.ndarray] = []
    label_segments: list[np.ndarray] = []

    for segment in split_at_gaps(times_s, gap_threshold_s):
        seg_t = times_s[segment]
        seg_values = values[segment]
        seg_labels = labels[segment]
        if len(seg_t) < 2:
            continue

        seg_t = seg_t - seg_t[0]
        duration = float(seg_t[-1] - seg_t[0])
        if duration <= 0:
            continue

        target_t = np.arange(0.0, duration, 1.0 / target_fs, dtype=np.float64)
        if len(target_t) == 0:
            continue

        fs = effective_fs(seg_t)
        filtered = lowpass_if_needed(seg_values, fs=fs, cutoff=cutoff, order=filter_order)
        resampled = np.column_stack([
            np.interp(target_t, seg_t, filtered[:, axis_idx])
            for axis_idx in range(filtered.shape[1])
        ])
        aligned_labels = nearest_labels(seg_t, seg_labels, target_t)

        signal_segments.append(resampled.astype(np.float32, copy=False))
        label_segments.append(aligned_labels.astype(np.int64, copy=False))

    if not signal_segments:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int64)

    return np.vstack(signal_segments), np.concatenate(label_segments)


def compute_summary_stats(values: np.ndarray, *, dataset_name: str, user: str, device: str, sensor: str) -> dict:
    stats = {
        "dataset": dataset_name,
        "User": user,
        "Device": device,
        "Sensor": f"watch_{sensor}",
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
            f"watch_{sensor}_{axis}_energy": float(np.sum(v ** 2) / len(v)),
        })
    return stats


def process_group(task: tuple) -> tuple[dict | None, pd.DataFrame | None, str]:
    (
        sensor,
        user,
        model,
        device,
        group_df,
        output_dir,
        dataset_name,
        target_fs,
        cutoff,
        filter_order,
        gap_threshold_s,
        min_samples,
    ) = task

    subject_key = f"{sanitize_token(user)}-{sanitize_token(device)}"
    signal, labels = resample_recording(
        group_df,
        target_fs=target_fs,
        cutoff=cutoff,
        filter_order=filter_order,
        gap_threshold_s=gap_threshold_s,
    )

    if len(signal) < min_samples:
        return None, None, f"skip {user}/{device}/{sensor}: only {len(signal)} output samples"

    duration_sec = len(signal) / target_fs
    base = f"{dataset_name}_{subject_key}_watch_{sensor}_{int(target_fs)}Hz_{duration_sec:.2f}s"
    output_path = Path(output_dir)
    pd.DataFrame(signal, columns=AXES).to_parquet(output_path / f"{base}.parquet", index=False)

    timestamp = np.arange(len(labels), dtype=np.float64) / target_fs
    label_df = pd.DataFrame({"timestamp": timestamp, "label": labels})
    label_df.to_parquet(output_path / f"{base}_labels.parquet", index=False)

    summary = compute_summary_stats(
        signal,
        dataset_name=dataset_name,
        user=str(user),
        device=str(device),
        sensor=sensor,
    )
    summary.update({
        "subject_key": subject_key,
        "Model": model,
        "duration_sec": duration_sec,
        "label_file": f"{base}_labels.parquet",
        "signal_file": f"{base}.parquet",
    })

    labels_summary = pd.DataFrame({
        "User": str(user),
        "Device": str(device),
        "subject_key": subject_key,
        "Sensor": f"watch_{sensor}",
        "timestamp": timestamp,
        "label": labels,
    })
    return summary, labels_summary, f"wrote {base}.parquet ({len(signal)} samples)"


def load_watch_tasks(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
    min_samples: int,
) -> list[tuple]:
    tasks = []
    files = {
        "accelerometer": input_dir / "Watch_accelerometer.csv",
        "gyroscope": input_dir / "Watch_gyroscope.csv",
    }

    usecols = ["Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"]
    grouped: dict[str, dict[tuple, pd.DataFrame]] = {}
    for sensor, path in files.items():
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue

        print(f"Loading {path}")
        df = pd.read_csv(path, usecols=usecols)
        grouped[sensor] = {
            key: group.reset_index(drop=True)
            for key, group in df.groupby(["User", "Model", "Device"], sort=True)
        }

    if set(grouped) != {"accelerometer", "gyroscope"}:
        return tasks

    common_keys = sorted(set(grouped["accelerometer"]) & set(grouped["gyroscope"]))
    skipped_missing = (set(grouped["accelerometer"]) ^ set(grouped["gyroscope"]))
    for user, model, device in sorted(skipped_missing):
        print(f"WARNING: skipping {user}/{device}: missing one watch sensor")

    for user, model, device in common_keys:
        acc_group = grouped["accelerometer"][(user, model, device)]
        gyro_group = grouped["gyroscope"][(user, model, device)]
        overlap_start = max(acc_group["Creation_Time"].min(), gyro_group["Creation_Time"].min())
        overlap_end = min(acc_group["Creation_Time"].max(), gyro_group["Creation_Time"].max())
        if overlap_end <= overlap_start:
            print(f"WARNING: skipping {user}/{device}: no acc/gyro overlap")
            continue

        sensor_groups = {
            "accelerometer": acc_group[
                (acc_group["Creation_Time"] >= overlap_start)
                & (acc_group["Creation_Time"] <= overlap_end)
            ].reset_index(drop=True),
            "gyroscope": gyro_group[
                (gyro_group["Creation_Time"] >= overlap_start)
                & (gyro_group["Creation_Time"] <= overlap_end)
            ].reset_index(drop=True),
        }
        for sensor, group in sensor_groups.items():
            tasks.append(
                (
                    sensor,
                    user,
                    model,
                    device,
                    group,
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


def plot_label_summary(labels_df: pd.DataFrame, output_dir: Path) -> None:
    dist = labels_df.groupby(["User", "Device", "Sensor", "label"]).size().reset_index(name="count")
    dist.to_csv(output_dir / "label_distribution_summary.csv", index=False)

    counts = labels_df["label"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar([str(label) for label in counts.index], counts.values)
    plt.title("HHAR Watch Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution_overall.png")
    plt.close()

    per_subject = labels_df.groupby(["subject_key", "label"]).size().reset_index(name="count")
    pivot = per_subject.pivot(index="subject_key", columns="label", values="count").fillna(0)
    plt.figure(figsize=(10, max(5, len(pivot) * 0.25)))
    plt.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar(label="Count")
    plt.xticks(np.arange(len(pivot.columns)), [str(col) for col in pivot.columns])
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.title("Label Counts per HHAR Watch Recording")
    plt.xlabel("Label")
    plt.ylabel("Subject-Device")
    plt.tight_layout()
    plt.savefig(output_dir / "label_heatmap.png")
    plt.close()


def parse_signal_name(path: Path) -> tuple[str, str, float] | None:
    match = re.match(
        r"HHAR_(.+)_watch_(accelerometer|gyroscope)_20Hz_([0-9.]+)s\.parquet$",
        path.name,
    )
    if not match:
        return None
    return match.group(1), match.group(2), float(match.group(3))


def harmonize_sensor_pairs(output_dir: Path, dataset_name: str, target_fs: float) -> None:
    """Make each subject-device's acc/gyro files the same length for loader pairing."""
    grouped: dict[str, dict[str, Path]] = {}
    for path in output_dir.glob(f"{dataset_name}_*_watch_*.parquet"):
        if path.name.endswith("_labels.parquet"):
            continue
        parsed = parse_signal_name(path)
        if parsed is None:
            continue
        subject_key, sensor, _duration = parsed
        grouped.setdefault(subject_key, {})[sensor] = path

    for subject_key, files in grouped.items():
        if set(files) != {"accelerometer", "gyroscope"}:
            continue

        lengths = {}
        for sensor, signal_path in files.items():
            lengths[sensor] = len(pd.read_parquet(signal_path, columns=["x"]))
        min_len = min(lengths.values())
        if min_len <= 0:
            continue

        duration_sec = min_len / target_fs
        for sensor, signal_path in list(files.items()):
            label_path = signal_path.with_name(signal_path.stem + "_labels.parquet")
            signal_df = pd.read_parquet(signal_path).iloc[:min_len].reset_index(drop=True)
            label_df = pd.read_parquet(label_path).iloc[:min_len].reset_index(drop=True)
            label_df["timestamp"] = np.arange(min_len, dtype=np.float64) / target_fs

            new_base = f"{dataset_name}_{subject_key}_watch_{sensor}_{int(target_fs)}Hz_{duration_sec:.2f}s"
            new_signal = output_dir / f"{new_base}.parquet"
            new_label = output_dir / f"{new_base}_labels.parquet"

            if signal_path != new_signal:
                signal_path.unlink()
            if label_path != new_label:
                label_path.unlink()

            signal_df.to_parquet(new_signal, index=False)
            label_df.to_parquet(new_label, index=False)


def collect_output_metadata(output_dir: Path, dataset_name: str, target_fs: float) -> tuple[list[dict], pd.DataFrame]:
    summaries = []
    label_frames = []
    for signal_path in sorted(output_dir.glob(f"{dataset_name}_*_watch_*.parquet")):
        if signal_path.name.endswith("_labels.parquet"):
            continue
        parsed = parse_signal_name(signal_path)
        if parsed is None:
            continue
        subject_key, sensor, duration_sec = parsed
        user, _, device = subject_key.partition("-")
        signal = pd.read_parquet(signal_path)[AXES].to_numpy(dtype=np.float32)
        label_path = signal_path.with_name(signal_path.stem + "_labels.parquet")
        labels = pd.read_parquet(label_path)

        summary = compute_summary_stats(
            signal,
            dataset_name=dataset_name,
            user=user,
            device=device,
            sensor=sensor,
        )
        summary.update({
            "subject_key": subject_key,
            "Model": re.sub(r"\d+$", "", device),
            "duration_sec": duration_sec,
            "label_file": label_path.name,
            "signal_file": signal_path.name,
        })
        summaries.append(summary)

        label_frame = labels.copy()
        label_frame.insert(0, "Sensor", f"watch_{sensor}")
        label_frame.insert(0, "subject_key", subject_key)
        label_frame.insert(0, "Device", device)
        label_frame.insert(0, "User", user)
        label_frames.append(label_frame)

    labels_df = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    return summaries, labels_df


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    target_fs: float,
    cutoff: float,
    filter_order: int,
    gap_threshold_s: float,
    min_samples: int,
    num_workers: int,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = load_watch_tasks(
        Path(input_dir),
        output_path,
        dataset_name,
        target_fs,
        cutoff,
        filter_order,
        gap_threshold_s,
        min_samples,
    )
    print(f"Processing {len(tasks)} watch sensor recordings with {num_workers} workers")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_group, task) for task in tasks]
        for future in as_completed(futures):
            summary, labels, message = future.result()
            print(message)

    harmonize_sensor_pairs(output_path, dataset_name=dataset_name, target_fs=target_fs)
    summaries, labels_df = collect_output_metadata(output_path, dataset_name=dataset_name, target_fs=target_fs)

    if summaries:
        pd.DataFrame(summaries).sort_values(["User", "Device", "Sensor"]).to_csv(
            output_path / f"{dataset_name}_summary_stats.csv",
            index=False,
        )

    if not labels_df.empty:
        labels_df.to_csv(output_path / f"{dataset_name}_all_labels.csv", index=False)
        plot_label_summary(labels_df, output_path)

    print(f"Completed HHAR watch preprocessing: {len(summaries)} signal files")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess HHAR watch data per user-device recording at 20 Hz."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="HHAR")
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