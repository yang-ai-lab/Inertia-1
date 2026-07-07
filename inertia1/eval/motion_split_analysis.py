from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import inspect

import importlib.util


def _load_motion_module(py_path: str):
    py_path = str(py_path)
    spec = importlib.util.spec_from_file_location("motion_dataloader", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load motion dataloader module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Create frozen MotionDataset splits and plot label distributions")
    p.add_argument("--data_root", required=True)
    default_dataloader = str(Path(__file__).resolve().parents[2] / "motion_dataloader.py")
    p.add_argument(
        "--dataloader_py",
        default=default_dataloader,
        help=(
            "Path to motion_dataloader.py (defaults to repo root motion_dataloader.py). "
            "If you pass a relative path, it is resolved against the current working directory."
        ),
    )
    p.add_argument("--out_dir", required=True)

    p.add_argument("--sampling_rate", type=float, default=20.0)
    p.add_argument("--window_sec", type=int, default=60)
    p.add_argument("--sensor_types", default="accelerometer")
    p.add_argument("--axial_mode", default="triaxial", choices=["triaxial", "uniaxial"])
    p.add_argument("--mode", default="nonoverlap", choices=["overlap", "nonoverlap"])
    p.add_argument(
        "--overlap_stride_samples",
        type=int,
        default=1,
        help="Stride in samples when --mode overlap is used.",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--group_by_subject", action="store_true")
    p.add_argument(
        "--group_key",
        default="subject_id",
        help="When using --group_by_subject, group windows by this MotionDataset sample metadata key (e.g., subject_id or session_id)",
    )
    p.add_argument(
        "--group_split_restarts",
        type=int,
        default=200,
        help="Number of randomized greedy restarts for subject-grouped splitting (higher is slower but can balance better)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing {dataset}_splits.json files")

    p.add_argument(
        "--ensure_label_coverage",
        action="store_true",
        help="When using --group_by_subject, try to ensure each (feasible) label appears in train/val/test by moving whole subjects",
    )
    p.add_argument(
        "--strict_label_coverage",
        action="store_true",
        help="Error out if label coverage cannot be satisfied (only applies with --ensure_label_coverage)",
    )
    p.add_argument(
        "--min_groups_per_label",
        type=int,
        default=3,
        help="Labels present in fewer than this many subjects are considered impossible to cover in all splits without leakage",
    )

    p.add_argument(
        "--max_unwanted_frac",
        type=float,
        default=0.5,
        help="Max fraction of unwanted labels allowed within a window before dropping it (larger keeps more windows)",
    )

    p.add_argument(
        "--allowed_label_ids",
        type=int,
        nargs="*",
        default=None,
        help="If set, keep only these encoded label ids; all others are mapped to -1 and may be filtered",
    )

    p.add_argument("--only_datasets", nargs="*", default=None)
    p.add_argument("--max_files_per_dataset", type=int, default=None)
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--subsample_ratio", type=float, default=None)

    p.add_argument(
        "--top_k_labels",
        type=int,
        default=None,
        help="If set, keep only the top-K most frequent labels (excluding unwanted) and remap them to 0..K-1",
    )

    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataloader_py = str(Path(args.dataloader_py).expanduser().resolve())
    print(f"Loading motion dataloader from: {dataloader_py}")
    motion = _load_motion_module(dataloader_py)

    dataset_keys = list(getattr(motion, "DATASET_CONFIGS").keys())
    if args.only_datasets:
        wanted = set(args.only_datasets)
        dataset_keys = [k for k in dataset_keys if k in wanted]

    window_size = int(round(args.sampling_rate * args.window_sec))

    for key in dataset_keys:
        print(f"\n=== {key} ===")
        try:
            ds_kwargs = dict(
                data_root=args.data_root,
                dataset_name=key,
                sampling_rate=float(args.sampling_rate),
                window_size=window_size,
                sensor_types=args.sensor_types,
                axial_mode=args.axial_mode,
                mode=str(args.mode),
                overlap_stride_samples=int(args.overlap_stride_samples),
                preload=False,
                return_majority_label=True,
                top_k_labels=args.top_k_labels,
                max_unwanted_frac=float(args.max_unwanted_frac),
                max_files_per_dataset=args.max_files_per_dataset,
                max_windows=args.max_windows,
                subsample_ratio=args.subsample_ratio,
            )

            # Backwards/forwards compatibility: some MotionDataset versions don't support allowed_label_ids.
            if args.allowed_label_ids is not None:
                sig = inspect.signature(motion.MotionDataset.__init__)
                if "allowed_label_ids" in sig.parameters:
                    ds_kwargs["allowed_label_ids"] = args.allowed_label_ids
                else:
                    raise TypeError(
                        "--allowed_label_ids was provided, but motion_dataloader.MotionDataset does not support it. "
                        "Remove --allowed_label_ids or update MotionDataset to accept it."
                    )

            ds = motion.MotionDataset(**ds_kwargs)
        except Exception as e:
            print(f"  ERROR creating dataset: {e}")
            continue

        if len(ds) == 0:
            print("  Skipping: no valid windows")
            continue

        # Compute true labels once for plotting.
        ys, _groups = motion.compute_window_labels_and_groups(ds)

        split_path = out_dir / f"{key}_splits.json"
        splits = motion.make_or_load_frozen_split(
            ds,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            group_by_subject=bool(args.group_by_subject),
            group_key=str(args.group_key),
            group_split_restarts=int(args.group_split_restarts),
            ensure_label_coverage=bool(args.ensure_label_coverage),
            strict_label_coverage=bool(args.strict_label_coverage),
            min_groups_per_label=int(args.min_groups_per_label),
            split_path=str(split_path),
            overwrite=bool(args.overwrite),
        )

        # Map indices to display labels if available.
        idx_to_label: Dict[int, str] = {}
        if getattr(ds, "idx_to_label", None):
            idx_to_label = {int(k): str(v) for k, v in ds.idx_to_label.items()}

        plot_path = out_dir / f"{key}_label_distribution.png"
        try:
            motion.plot_split_label_distributions(
                ys,
                splits,
                title=f"{key} label distribution (train/val/test)",
                save_path=str(plot_path),
                idx_to_label=idx_to_label or None,
            )
            print(f"  Wrote {plot_path}")
        except Exception as e:
            print(f"  Plot skipped: {e}")


if __name__ == "__main__":
    main()
