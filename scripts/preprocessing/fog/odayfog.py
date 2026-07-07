#!/usr/bin/env python3
"""Rebuild the audited OdayFoG processed subset for downstream evaluation.

This script follows `scripts/preprocessing/DATASET_AUDIT_PLAYBOOK.md` for
`OdayFoG`:
- raw recordings live as one Excel workbook per walk trial under `data/raw/`;
- the dataset is binary FoG detection with raw `freeze_label` values mapped to
  `{0, 1}` and missing labels mapped to `-1`;
- signals are rebuilt at 20 Hz from the raw 128 Hz timeline with a 4th-order
  10 Hz Butterworth low-pass filter followed by time-based interpolation;
- labels are rebuilt by nearest-neighbor assignment onto the 20 Hz target
  timeline;
- only placements that are actually present in a raw workbook are written;
- when the same recording exists in both `imus6_subjects7` and
  `imus11_subjects4`, the richer `imus11_subjects4` copy is preferred;
- the current processed naming drops the raw `walklr` token. Per user choice we
  keep that filename convention, so the one collided recording
  (`subject7_v12_t2`) keeps only the branch that matches the existing shared
  label file and drops the unmatched branch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./data/raw/OdayFoG"))
RAW_ROOT = DATASET_ROOT / "data" / "raw"
SOURCE_PROCESSED_ROOT = DATASET_ROOT / "processed"
OUTPUT_ROOT = DATASET_ROOT / "processed_rebuild_tmp"

TARGET_FS = 20.0
FILTER_CUTOFF_HZ = 10.0
FILTER_ORDER = 4
GAP_THRESHOLD_S = 1.0
MANIFEST_NAME = "odayfog_rebuild_manifest.json"

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
RID_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
TIME_COLUMN_RE = re.compile(rb'<c[^>]*\sr="B\d+"[^>]*><v>([^<]+)</v></c>')

AXES = ("x", "y", "z")
SENSOR_SPECS = {
    "accelerometer": ("ax", "ay", "az"),
    "gyroscope": ("gx", "gy", "gz"),
}
PLACEMENTS = (
    "head",
    "chest",
    "lumbar",
    "wrist_r",
    "wrist_l",
    "thigh_r",
    "thigh_l",
    "ankle_r",
    "ankle_l",
    "foot_r",
    "foot_l",
)
RAW_NAME_RE = re.compile(
    r"^pt(?P<subject>\d+)_visit_(?P<visit>[0-9.]+)_tbc_walklr_(?P<walk>[01])_trial_(?P<trial>\d+)$"
)


@dataclass(frozen=True)
class CandidateRecording:
    record_id: str
    walk_token: str
    path: Path


@dataclass(frozen=True)
class SelectedRecording:
    record_id: str
    path: Path
    walk_token: str
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--source-processed-root", type=Path, default=SOURCE_PROCESSED_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument(
        "--recording-ids",
        nargs="*",
        default=None,
        help="Optional list of collapsed processed recording ids to rebuild.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default=MANIFEST_NAME,
        help="Manifest filename written inside the output directory.",
    )
    return parser.parse_args()


def normalize_target(target: str) -> str:
    target = target.lstrip("/")
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return target.replace("xl//", "xl/")


def workbook_sheet_target(zf: ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("rel:Relationship", NS)}
    sheet = workbook.find("main:sheets/main:sheet", NS)
    if sheet is None:
        raise ValueError("Workbook does not contain any sheets")
    return normalize_target(rel_map[sheet.attrib[RID_NS]])


def load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall("main:si", NS):
        texts = [t.text or "" for t in si.findall(".//main:t", NS)]
        strings.append("".join(texts))
    return strings


def col_letters(cell_ref: str) -> str:
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch)
        else:
            break
    return "".join(letters)


def safe_float(value: object) -> float:
    if value is None or value == "":
        return np.nan
    return float(value)


def parse_label(value: object) -> int:
    if value is None or value == "":
        return -1
    return int(round(float(value)))


def remove_existing_contents(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def parse_raw_name(path: Path) -> CandidateRecording:
    match = RAW_NAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"Unexpected raw filename: {path.name}")
    subject = f"subject{match.group('subject')}"
    visit = f"v{match.group('visit').replace('.', '-')}"
    trial = f"t{match.group('trial')}"
    record_id = f"OdayFoG_{subject}_{visit}_{trial}"
    return CandidateRecording(record_id=record_id, walk_token=match.group("walk"), path=path)


def source_priority(path: Path) -> tuple[int, str]:
    if path.parent.name == "imus11_subjects4":
        return (0, path.name)
    return (1, path.name)


def load_existing_label_len(processed_root: Path, record_id: str) -> int | None:
    path = processed_root / f"{record_id}_labels.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    series = df.iloc[:, -1]
    if len(series) == 1 and isinstance(series.iloc[0], (list, np.ndarray)):
        return int(len(series.iloc[0]))
    return int(len(series))


def estimate_target_len(path: Path) -> int:
    with ZipFile(path) as zf:
        data = zf.read(workbook_sheet_target(zf))
    times = [float(x.decode("utf-8")) for x in TIME_COLUMN_RE.findall(data)[1:]]
    if len(times) < 2:
        return 0
    duration = times[-1] - times[0]
    return int(len(np.arange(times[0], times[-1], 1.0 / TARGET_FS, dtype=np.float64)))


def select_recordings(raw_root: Path, processed_root: Path) -> tuple[list[SelectedRecording], list[dict]]:
    grouped: dict[str, list[CandidateRecording]] = {}
    for path in sorted(raw_root.rglob("*.xlsx")):
        candidate = parse_raw_name(path)
        grouped.setdefault(candidate.record_id, []).append(candidate)

    selected: list[SelectedRecording] = []
    collisions: list[dict] = []
    for record_id in sorted(grouped):
        candidates = grouped[record_id]
        walk_groups: dict[str, list[CandidateRecording]] = {}
        for candidate in candidates:
            walk_groups.setdefault(candidate.walk_token, []).append(candidate)

        preferred_by_walk = {
            walk: sorted(items, key=lambda item: source_priority(item.path))[0]
            for walk, items in walk_groups.items()
        }
        if len(preferred_by_walk) == 1:
            only = next(iter(preferred_by_walk.values()))
            selected.append(
                SelectedRecording(
                    record_id=record_id,
                    path=only.path,
                    walk_token=only.walk_token,
                    selection_reason="prefer_imus11_if_duplicate",
                )
            )
            continue

        existing_label_len = load_existing_label_len(processed_root, record_id)
        if existing_label_len is None:
            raise ValueError(
                f"Collapsed recording {record_id} has multiple raw walklr variants but no existing label file "
                "to disambiguate them."
            )

        matches = []
        candidate_lengths: dict[str, int] = {}
        for walk, candidate in preferred_by_walk.items():
            target_len = estimate_target_len(candidate.path)
            candidate_lengths[walk] = target_len
            if target_len == existing_label_len:
                matches.append(candidate)

        if len(matches) != 1:
            raise ValueError(
                f"Could not disambiguate collapsed recording {record_id}: "
                f"existing label length={existing_label_len}, candidate lengths={candidate_lengths}"
            )

        chosen = matches[0]
        dropped = [candidate.path.name for walk, candidate in preferred_by_walk.items() if walk != chosen.walk_token]
        selected.append(
            SelectedRecording(
                record_id=record_id,
                path=chosen.path,
                walk_token=chosen.walk_token,
                selection_reason="match_existing_label_length_under_kept_naming",
            )
        )
        collisions.append(
            {
                "record_id": record_id,
                "kept_walklr": chosen.walk_token,
                "kept_source": str(chosen.path),
                "existing_label_len": existing_label_len,
                "candidate_target_lengths": candidate_lengths,
                "dropped_sources": dropped,
            }
        )
    return selected, collisions


def load_workbook_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, str], np.ndarray]]:
    wanted_headers = {"time", "freeze_label"}
    for placement in PLACEMENTS:
        for suffixes in SENSOR_SPECS.values():
            for suffix in suffixes:
                wanted_headers.add(f"imu_{placement}_{suffix}")

    with ZipFile(path) as zf:
        shared = load_shared_strings(zf)
        worksheet = ET.fromstring(zf.read(workbook_sheet_target(zf)))

    header_map: dict[str, str] | None = None
    active_letters: dict[str, str] = {}
    values_by_header: dict[str, list[object]] = {}
    for row in worksheet.findall("main:sheetData/main:row", NS):
        row_values: dict[str, object] = {}
        for cell in row.findall("main:c", NS):
            ref = cell.attrib.get("r", "")
            letter = col_letters(ref)
            cell_type = cell.attrib.get("t")
            value_node = cell.find("main:v", NS)
            inline_node = cell.find("main:is", NS)
            if cell_type == "s" and value_node is not None:
                value: object = shared[int(value_node.text)]
            elif cell_type == "inlineStr" and inline_node is not None:
                value = "".join(t.text or "" for t in inline_node.findall(".//main:t", NS))
            elif value_node is not None:
                value = value_node.text
            else:
                value = None
            row_values[letter] = value

        if header_map is None:
            header_map = {letter: str(value) for letter, value in row_values.items() if value in wanted_headers}
            active_letters = {letter: header for letter, header in header_map.items() if header in wanted_headers}
            values_by_header = {header: [] for header in active_letters.values()}
            continue

        for letter, header in active_letters.items():
            values_by_header[header].append(row_values.get(letter))

    if not values_by_header:
        raise ValueError(f"Failed to parse workbook rows from {path}")

    times = np.asarray([safe_float(v) for v in values_by_header["time"]], dtype=np.float64)
    labels = np.asarray([parse_label(v) for v in values_by_header["freeze_label"]], dtype=np.int64)
    stream_arrays: dict[tuple[str, str], np.ndarray] = {}
    for placement in PLACEMENTS:
        for sensor_name, suffixes in SENSOR_SPECS.items():
            cols = [f"imu_{placement}_{suffix}" for suffix in suffixes]
            if any(col not in values_by_header for col in cols):
                continue
            arr = np.column_stack(
                [np.asarray([safe_float(v) for v in values_by_header[col]], dtype=np.float64) for col in cols]
            )
            if np.isnan(arr).all():
                continue
            stream_arrays[(placement, sensor_name)] = arr

    valid_time_mask = np.isfinite(times)
    if not valid_time_mask.any():
        raise ValueError(f"No valid timestamps found in {path}")

    times = times[valid_time_mask]
    labels = labels[valid_time_mask]
    stream_arrays = {key: value[valid_time_mask] for key, value in stream_arrays.items()}

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    labels = labels[order]
    stream_arrays = {key: value[order] for key, value in stream_arrays.items()}

    unique_times, unique_indices = np.unique(times, return_index=True)
    times = unique_times
    labels = labels[unique_indices]
    stream_arrays = {key: value[unique_indices] for key, value in stream_arrays.items()}
    return times, labels, stream_arrays


def split_at_gaps(times_s: np.ndarray, gap_threshold_s: float = GAP_THRESHOLD_S) -> list[slice]:
    if len(times_s) == 0:
        return []
    gap_idx = np.flatnonzero(np.diff(times_s) > gap_threshold_s) + 1
    starts = np.r_[0, gap_idx]
    ends = np.r_[gap_idx, len(times_s)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def effective_fs(times_s: np.ndarray) -> float:
    if len(times_s) < 2:
        return 0.0
    duration = float(times_s[-1] - times_s[0])
    if duration <= 0:
        return 0.0
    return float((len(times_s) - 1) / duration)


def lowpass_if_needed(values: np.ndarray, fs: float) -> np.ndarray:
    if fs <= TARGET_FS or len(values) <= (FILTER_ORDER * 3 + 1):
        return values
    nyq = 0.5 * fs
    cutoff = min(FILTER_CUTOFF_HZ, np.nextafter(nyq, 0.0))
    if cutoff <= 0:
        return values
    b, a = butter(FILTER_ORDER, cutoff / nyq, btype="low", analog=False)
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


def resample_stream(
    record_times: np.ndarray,
    raw_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    valid_mask = np.isfinite(raw_values).all(axis=1)
    stream_times = record_times[valid_mask]
    stream_values = raw_values[valid_mask]
    if len(stream_times) < 2:
        return np.empty((0, 3), dtype=np.float32)
    segments = split_at_gaps(stream_times)
    if len(segments) != 1:
        raise ValueError("Unexpected >1 s internal gap in OdayFoG stream")
    fs = effective_fs(stream_times)
    filtered = lowpass_if_needed(stream_values, fs=fs)
    resampled = np.column_stack(
        [np.interp(target_times, stream_times, filtered[:, axis_idx]) for axis_idx in range(filtered.shape[1])]
    )
    return resampled.astype(np.float32, copy=False)


def process_recording(task: tuple[SelectedRecording, Path]) -> dict:
    selected, output_root = task
    record_times, raw_labels, stream_arrays = load_workbook_arrays(selected.path)
    if len(record_times) < 2:
        raise ValueError(f"Recording {selected.record_id} has fewer than 2 valid timestamps")

    if len(split_at_gaps(record_times)) != 1:
        raise ValueError(f"Recording {selected.record_id} contains >1 s gaps unexpectedly")

    active_spans = []
    for stream in stream_arrays.values():
        valid_mask = np.isfinite(stream).all(axis=1)
        if not valid_mask.any():
            continue
        valid_times = record_times[valid_mask]
        active_spans.append((float(valid_times[0]), float(valid_times[-1])))
    if not active_spans:
        raise ValueError(f"Recording {selected.record_id} has no active sensor streams")

    common_start = max(span[0] for span in active_spans)
    common_end = min(span[1] for span in active_spans)
    target_times = np.arange(common_start, common_end, 1.0 / TARGET_FS, dtype=np.float64)
    if len(target_times) == 0:
        raise ValueError(f"Recording {selected.record_id} has no common 20 Hz overlap")

    rebuilt_streams: dict[tuple[str, str], np.ndarray] = {}
    for key, raw_values in stream_arrays.items():
        signal = resample_stream(record_times, raw_values, target_times)
        if len(signal) != len(target_times):
            raise ValueError(f"Signal/target length mismatch for {selected.record_id} {key}")
        rebuilt_streams[key] = signal

    labels = nearest_labels(record_times, raw_labels, target_times).astype(np.int64, copy=False)
    duration_s = len(target_times) / TARGET_FS
    duration_token = f"{duration_s:.2f}s"

    written_files = []
    for (placement, sensor_name), signal in sorted(rebuilt_streams.items()):
        sensor_token = "acc" if sensor_name == "accelerometer" else "gyro"
        filename = f"{selected.record_id}_{placement}_{sensor_token}_20Hz_{duration_token}.parquet"
        path = output_root / filename
        pd.DataFrame(signal, columns=list(AXES)).to_parquet(path, index=False)
        written_files.append(filename)

    label_path = output_root / f"{selected.record_id}_labels.parquet"
    label_df = pd.DataFrame(
        {
            "timestamp": np.arange(len(labels), dtype=np.float64) / TARGET_FS,
            "label": labels,
        }
    )
    label_df.to_parquet(label_path, index=False)

    label_counts = {}
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        label_counts[int(label)] = int(count)

    fs_estimate = effective_fs(record_times)
    return {
        "record_id": selected.record_id,
        "raw_path": str(selected.path),
        "walklr": selected.walk_token,
        "selection_reason": selected.selection_reason,
        "raw_rows": int(len(record_times)),
        "effective_fs": float(fs_estimate),
        "output_rows": int(len(target_times)),
        "duration_s": float(duration_s),
        "placements_written": sorted({placement for placement, _sensor in rebuilt_streams}),
        "sensor_types_written": sorted({sensor for _placement, sensor in rebuilt_streams}),
        "signal_file_count": int(len(written_files)),
        "label_counts": label_counts,
        "label_file": label_path.name,
        "signal_files": written_files,
    }


def filter_selected_recordings(
    recordings: Iterable[SelectedRecording],
    wanted_ids: set[str] | None,
) -> list[SelectedRecording]:
    filtered = list(recordings)
    if wanted_ids is None:
        return filtered
    return [recording for recording in filtered if recording.record_id in wanted_ids]


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if args.clean_output:
        remove_existing_contents(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected, collisions = select_recordings(args.raw_root, args.source_processed_root)
    wanted_ids = set(args.recording_ids) if args.recording_ids else None
    selected = filter_selected_recordings(selected, wanted_ids)
    if not selected:
        raise ValueError("No recordings selected for rebuild")

    summaries = []
    tasks = [(recording, output_root) for recording in selected]
    if int(args.workers) <= 1 or len(tasks) == 1:
        for task in tasks:
            summaries.append(process_recording(task))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            future_map = {executor.submit(process_recording, task): task[0].record_id for task in tasks}
            for future in as_completed(future_map):
                summaries.append(future.result())

    summaries.sort(key=lambda row: row["record_id"])
    manifest = {
        "dataset": "OdayFoG",
        "raw_root": str(args.raw_root),
        "source_processed_root": str(args.source_processed_root),
        "output_root": str(output_root),
        "target_fs": TARGET_FS,
        "filter_cutoff_hz": FILTER_CUTOFF_HZ,
        "filter_order": FILTER_ORDER,
        "selected_recordings": len(selected),
        "collision_resolution": collisions,
        "notes": [
            "Collapsed processed naming was kept per user request, so the lone walklr collision keeps only the branch that matches the existing shared label file.",
            "Signals are written only for placements/sensors that are truly present in the raw workbook; no all-None placeholder streams are emitted.",
            "Shared label parquets are rebuilt as [timestamp, label] at 20 Hz with nearest-neighbor assignment from the raw freeze_label timeline.",
        ],
        "recordings": summaries,
    }
    manifest_path = output_root / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_signals = sum(row["signal_file_count"] for row in summaries)
    print(
        f"Rebuilt {len(summaries)} recordings into {output_root} "
        f"({total_signals} signal files + {len(summaries)} shared labels)"
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
