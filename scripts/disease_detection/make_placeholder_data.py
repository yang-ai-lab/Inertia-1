"""Generate a tiny synthetic dataset for the disease-detection MIL pipeline.

This exists purely so the pipeline is runnable end-to-end without the
closed-source NHANES corpus. It writes:

    <out>/windows/<patient_id>.npy   # float array [N_windows, C, T]
    <out>/labels.csv                 # patient_id,label
    <out>/split.csv                  # patient_id,split  (train/val/test)

The "signal" is pure noise with a tiny class-dependent offset, so a probe can
fit but numbers are meaningless -- it is a smoke-test fixture, not real data.

Example:
    python scripts/disease_detection/make_placeholder_data.py \
        --out ./data/disease --num-patients 60 --num-classes 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./data/disease")
    p.add_argument("--num-patients", type=int, default=60)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--axes", type=int, default=3)
    p.add_argument("--window-sec", type=int, default=30)
    p.add_argument("--hz", type=int, default=20)
    p.add_argument("--min-windows", type=int, default=40)
    p.add_argument("--max-windows", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    win_dir = out / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    t = int(args.window_sec * args.hz)
    c = 1 if args.axes == 1 else 3

    labels, splits = [], []
    for i in range(args.num_patients):
        pid = f"P{i:04d}"
        y = int(rng.integers(0, args.num_classes))
        n = int(rng.integers(args.min_windows, args.max_windows + 1))
        windows = rng.standard_normal((n, c, t)).astype(np.float32) + 0.05 * y
        np.save(win_dir / f"{pid}.npy", windows)
        labels.append((pid, y))
        # 70/15/15 split
        r = rng.random()
        split = "train" if r < 0.7 else ("val" if r < 0.85 else "test")
        splits.append((pid, split))

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label"])
        w.writerows(labels)
    with open(out / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "split"])
        w.writerows(splits)

    print(f"Wrote {args.num_patients} patients to {win_dir}")
    print(f"Labels:  {out / 'labels.csv'}")
    print(f"Splits:  {out / 'split.csv'}")


if __name__ == "__main__":
    main()
