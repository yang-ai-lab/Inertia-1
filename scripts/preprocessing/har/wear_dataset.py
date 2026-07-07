#!/usr/bin/env python3
"""Prepare the audited WEAR processed subset for downstream evaluation.

This script follows the WEAR audit outcome from
`scripts/preprocessing/DATASET_AUDIT_PLAYBOOK.md`:
- the current processed signals/labels already match the intended implicit-50 Hz
  raw-to-20 Hz preprocessing for the in-scope recordings;
- the audited default downstream placement is `right_arm`;
- all 18 exercise labels are kept, with only raw `nan` rows treated as
  unwanted/null by the dataloader;
- one corrupted signal file,
  `WEAR_S10_left_arm_acc_20Hz_3985.00s.parquet`, is dropped entirely because the
  raw `sbj_10` left-arm stream contains a long contiguous NaN outage.

The script materializes `processed_rebuild_tmp` from the current production
`processed/` directory without touching production data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./data/raw/wear_dataset"))
CORRUPTED_SIGNAL = "WEAR_S10_left_arm_acc_20Hz_3985.00s.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DATASET_ROOT / "processed",
        help="Current production processed directory to copy from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "processed_rebuild_tmp",
        help="Temporary rebuild directory to populate.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATASET_ROOT / "IMUdata",
        help="Raw WEAR CSV directory, used only to record the corrupted span details.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of parallel file-copy workers.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing output files before copying the rebuilt subset.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="wear_rebuild_manifest.json",
        help="Audit manifest filename written inside the output directory.",
    )
    return parser.parse_args()


def remove_existing_contents(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def subject_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("WEAR_S"):
        remainder = stem.split("_", 2)
        if len(remainder) >= 2:
            try:
                return int(remainder[1][1:]), path.name
            except ValueError:
                pass
    return (10**9, path.name)


def copy_file(src: Path, dst: Path) -> dict:
    shutil.copy2(src, dst)
    return {
        "filename": src.name,
        "bytes": int(src.stat().st_size),
    }


def compute_raw_nan_manifest(raw_dir: Path) -> dict:
    raw_path = raw_dir / "sbj_10.csv"
    if not raw_path.exists():
        return {"raw_file": str(raw_path), "exists": False}

    cols = ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"]
    raw = pd.read_csv(raw_path, usecols=cols, engine="python")
    mask = raw.isna().any(axis=1).to_numpy()
    if not mask.any():
        return {
            "raw_file": str(raw_path),
            "exists": True,
            "nan_rows": 0,
            "nan_runs": [],
        }

    changes = (mask[1:] != mask[:-1]).nonzero()[0] + 1
    starts = [0, *changes.tolist()]
    ends = [*changes.tolist(), len(mask)]
    runs = []
    for start, end in zip(starts, ends):
        if not mask[start]:
            continue
        runs.append(
            {
                "start_row": int(start),
                "end_row_exclusive": int(end),
                "n_rows": int(end - start),
                "start_time_s": float(start / 50.0),
                "end_time_s": float(end / 50.0),
                "duration_s": float((end - start) / 50.0),
            }
        )

    return {
        "raw_file": str(raw_path),
        "exists": True,
        "nan_rows": int(mask.sum()),
        "nan_runs": runs,
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        remove_existing_contents(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        [p for p in source_dir.iterdir() if p.is_file()],
        key=subject_sort_key,
    )
    kept_files = [p for p in source_files if p.name != CORRUPTED_SIGNAL]
    dropped_files = [p.name for p in source_files if p.name == CORRUPTED_SIGNAL]

    copied = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {
            executor.submit(copy_file, src, output_dir / src.name): src.name
            for src in kept_files
        }
        for future in as_completed(future_map):
            copied.append(future.result())

    copied.sort(key=lambda row: row["filename"])

    manifest = {
        "dataset": "wear_dataset",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "audited_default_placement": "right_arm",
        "kept_label_policy": "all_18_labels_keep_nan_as_unwanted_only",
        "copied_file_count": len(copied),
        "dropped_file_count": len(dropped_files),
        "dropped_files": dropped_files,
        "raw_corruption_detail": compute_raw_nan_manifest(args.raw_dir),
        "notes": [
            "WEAR raw CSVs use an implicit regular 50 Hz timeline and do not expose explicit timestamps.",
            "The processed WEAR signals/labels were retained as-is except for dropping the corrupted S10 left-arm signal file.",
            "Regenerate frozen split JSONs against processed_rebuild_tmp with placement=right_arm after running this script.",
        ],
    }

    manifest_path = output_dir / args.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Copied {len(copied)} files into {output_dir}")
    if dropped_files:
        print(f"Dropped corrupted files: {dropped_files}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
