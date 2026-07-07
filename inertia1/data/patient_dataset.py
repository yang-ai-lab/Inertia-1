from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _stable_hash_to_int(s: str, *, seed: int = 0) -> int:
    h = hashlib.md5(f"{seed}:{s}".encode("utf-8")).hexdigest()
    return int(h, 16)


def load_patient_split_csv(split_csv: str | Path) -> dict[str, list[str]]:
    """Load a CSV with columns: patient_id, split.

    Returns dict with keys like "train", "val".
    """
    split_csv = Path(split_csv)
    splits: dict[str, list[str]] = {}
    with split_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"patient_id", "split"}.issubset(set(reader.fieldnames)):
            raise ValueError(f"{split_csv} must have columns patient_id,split")
        for row in reader:
            pid = str(row["patient_id"]).strip()
            sp = str(row["split"]).strip().lower()
            if not pid or not sp:
                continue
            splits.setdefault(sp, []).append(pid)

    # Make patient ordering deterministic across ranks.
    for k in list(splits.keys()):
        splits[k] = sorted(set(splits[k]))

    return splits


def _infer_patient_id(root: Path, path: Path) -> str:
    """Infer patient ID from a file path.

    Expected layouts:
      - root/<patient_id>/<file>.parquet
      - root/<patient_id>/**/<file>.parquet

    If relative path has no parts (unexpected), fallback to parent directory name.
    """
    try:
        rel = path.relative_to(root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    return path.parent.name


def _is_rank_zero_process() -> bool:
    """Best-effort rank-0 check for DDP/SLURM.

    Keeps dataset init logs from printing 4x under DDP.
    """
    for k in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is not None and str(v).strip() != "0":
            return False
    return True


@dataclass(frozen=True)
class _PatientFileRange:
    pid: str
    file_start: int
    file_count: int
    group_starts: tuple[int, ...]
    sample_start: int
    sample_count: int


_FILENAME_DT_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")


def _extract_datetime_from_path(path: Path) -> Optional[_dt.datetime]:
    """Parse a timestamp embedded in filenames like:

    <prefix>_<subject_id>_YYYY-MM-DD-HH-MM-SS.parquet
    """

    m = _FILENAME_DT_RE.search(path.name)
    if not m:
        return None
    try:
        return _dt.datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S")
    except Exception:
        return None


class PatientGroupedAccelWindowDataset(Dataset):
    """Map-style dataset of fixed-duration windowed accelerometer data, grouped by patient.

    Key property: does NOT materialize an index of all windows.
    Instead:
      - stores a sorted file list [N_files]
      - stores per-patient contiguous file ranges [N_patients]
      - computes window selection within a file via arithmetic

    Global indices are contiguous per patient, enabling patient-level samplers.

    Requires assume_fixed_duration=True.
    """

    def __init__(
        self,
        *,
        data_root: str | Path,
        split_csv: str | Path,
        split: Literal["train", "val", "test"],
        file_extension: str = ".parquet",
        input_format: Literal["CT", "TC"] = "CT",
        parquet_columns: Optional[list[str]] = None,
        max_files: Optional[int] = None,
        # windowing
        window_size: int,
        stride: int,
        hz: float = 20.0,
        native_hz: float = 20.0,
        file_duration_sec: int = 600,
        assume_fixed_duration: bool = True,
        tolerate_bad_files: bool = True,
        bad_file_max_retries: int = 20,
        cache_last_file: bool = True,
        # axes
        axes: int = 3,
        axis_reduce: Literal["l2", "mean", "sum"] = "l2",
        concat_files: int = 1,
        require_consecutive_files: bool = True,
        return_id: bool = False,
        seed: int = 42,
        patient_subsample_ratio: float = 1.0,
        allowed_patient_ids: Optional[set[str]] = None,
    ):
        self.file_extension = file_extension
        self.input_format = input_format
        self.parquet_columns = parquet_columns or ["X", "Y", "Z"]
        self.hz = float(hz)
        self.native_hz = float(native_hz)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.file_duration_sec = int(file_duration_sec)
        self.assume_fixed_duration = bool(assume_fixed_duration)
        self.tolerate_bad_files = bool(tolerate_bad_files)
        self.bad_file_max_retries = int(bad_file_max_retries)
        self.cache_last_file = bool(cache_last_file)
        self.axes = int(axes)
        self.axis_reduce = axis_reduce
        self.concat_files = int(concat_files)
        self.require_consecutive_files = bool(require_consecutive_files)
        self.return_id = bool(return_id)
        self.seed = int(seed)
        self.patient_subsample_ratio = float(patient_subsample_ratio)
        self.allowed_patient_ids = allowed_patient_ids

        if self.concat_files <= 0:
            raise ValueError(f"concat_files must be >= 1, got {self.concat_files}")

        if not self.assume_fixed_duration:
            raise ValueError("PatientGroupedAccelWindowDataset currently requires assume_fixed_duration=true")
        if self.hz <= 0 or self.native_hz <= 0:
            raise ValueError(f"Invalid hz/native_hz: {self.hz}/{self.native_hz}")
        if self.hz > self.native_hz:
            raise ValueError(f"Target hz ({self.hz}) cannot exceed native_hz ({self.native_hz})")
        ratio = float(self.native_hz) / float(self.hz)
        ratio_rounded = int(round(ratio))
        if abs(ratio - ratio_rounded) > 1e-6:
            raise ValueError(
                f"native_hz ({self.native_hz}) must be evenly divisible by target hz ({self.hz}). "
                f"Ratio is {ratio}"
            )

        self.downsample_factor = ratio_rounded
        self._native_window_size = self.window_size * self.downsample_factor
        self._native_stride = self.stride * self.downsample_factor

        self._file_len_T = round(self.file_duration_sec * self.native_hz)
        if self._file_len_T <= 0:
            raise ValueError("file_duration_sec * native_hz must be > 0")

        # windows per concatenated group (fixed-duration assumption)
        self._group_len_T = int(self._file_len_T) * int(self.concat_files)
        last_start = self._group_len_T - self._native_window_size
        if last_start < 0:
            raise ValueError(
                "No windows per file. Check window_size/stride vs file_duration_sec*hz. "
                f"Got group_len_T={self._group_len_T}, window_size={self.window_size}, stride={self.stride}."
            )
        self._nwin_per_group = 1 + (last_start // self._native_stride)

        # patients to include
        split_map = load_patient_split_csv(split_csv)
        if split not in split_map:
            raise ValueError(f"split '{split}' not found in {split_csv}. Available: {sorted(split_map.keys())}")
        patient_ids = split_map[split]
        if self.allowed_patient_ids is not None:
            patient_ids = [pid for pid in patient_ids if pid in self.allowed_patient_ids]
            if not patient_ids:
                raise ValueError(
                    f"No patients remain after allowed_patient_ids filter for split='{split}'. "
                    f"Check labels vs split_csv."
                )
        if self.patient_subsample_ratio < 1.0:
            if not (0.0 < self.patient_subsample_ratio <= 1.0):
                raise ValueError(
                    f"patient_subsample_ratio must be in (0,1], got {self.patient_subsample_ratio}"
                )
            n_keep = max(1, int(round(len(patient_ids) * self.patient_subsample_ratio)))
            # Deterministic subsample independent of filesystem ordering.
            ranked = sorted(patient_ids, key=lambda pid: _stable_hash_to_int(pid, seed=self.seed))
            patient_ids = sorted(ranked[:n_keep])

        self.patient_ids = patient_ids
        patient_set = set(self.patient_ids)

        # resolve roots
        roots: list[Path]
        data_root_str = str(data_root)
        if "," in data_root_str:
            roots = [Path(p.strip()) for p in data_root_str.split(",")]
        else:
            roots = [Path(data_root_str)]

        # build file list filtered by patient
        records: list[tuple[str, Optional[_dt.datetime], Path]] = []
        t0 = time.time()
        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"data_root does not exist: {root}")
            if _is_rank_zero_process():
                print(
                    f"[ts_ssl][data] split={split} scanning root={root} "
                    f"patients={len(self.patient_ids):,} ext={self.file_extension}",
                    flush=True,
                )

            # Fast path: scan only the patient directories listed in split_csv.
            # This avoids a costly full-tree glob over the entire corpus.
            for pid in self.patient_ids:
                pid_dir = root / pid
                if not pid_dir.exists():
                    continue
                for path in pid_dir.rglob(f"*{self.file_extension}"):
                    records.append((pid, _extract_datetime_from_path(path), path))
                    if max_files is not None and len(records) >= int(max_files):
                        break
                if max_files is not None and len(records) >= int(max_files):
                    break

            if max_files is not None and len(records) >= int(max_files):
                break

        if max_files is not None:
            records = records[: int(max_files)]

        if _is_rank_zero_process():
            dt = time.time() - t0
            print(f"[ts_ssl][data] split={split} found_files={len(records):,} in {dt:.1f}s", flush=True)

        if not records:
            raise ValueError(
                f"No {self.file_extension} files found under {data_root} for split='{split}'. "
                f"Check split_csv={split_csv} and directory layout."
            )

        # sort by patient then parsed datetime (if available) then path so each patient's files are contiguous
        records.sort(key=lambda t: (t[0], t[1] or _dt.datetime.min, str(t[2])))
        self.file_paths = [p for _, _, p in records]
        self.file_datetimes = [dt for _, dt, _ in records]

        # build per-patient file ranges and global sample ranges
        self.patient_ranges: list[_PatientFileRange] = []
        sample_cursor = 0
        i = 0
        while i < len(records):
            pid = records[i][0]
            j = i + 1
            while j < len(records) and records[j][0] == pid:
                j += 1
            file_start = i
            file_count = j - i

            group_starts: list[int] = []
            if self.concat_files == 1:
                # Every file can act as its own group start.
                group_starts = [file_start + k for k in range(file_count)]
            else:
                dts = self.file_datetimes[file_start : file_start + file_count]
                if self.require_consecutive_files and any(dt is None for dt in dts):
                    # Without a monotonic timestamp/index, we cannot guarantee consecutiveness.
                    missing = next(
                        (self.file_paths[file_start + k] for k, dt in enumerate(dts) if dt is None),
                        None,
                    )
                    raise ValueError(
                        "require_consecutive_files=true but could not parse timestamps from some filenames. "
                        f"Example file: {missing}"
                    )

                # Valid start iff the next concat_files-1 deltas equal file_duration_sec.
                # This ensures the concatenated block represents continuous time with no gaps.
                ok_steps: list[bool] = []
                for k in range(file_count - 1):
                    if dts[k] is None or dts[k + 1] is None:
                        ok_steps.append(False)
                        continue
                    dt_sec = (dts[k + 1] - dts[k]).total_seconds()
                    ok_steps.append(abs(dt_sec - float(self.file_duration_sec)) <= 1e-6)

                need = self.concat_files - 1
                run = 0
                for step_i, ok in enumerate(ok_steps):
                    run = (run + 1) if ok else 0
                    if run >= need:
                        start_offset = step_i - (need - 1)
                        start_idx = file_start + start_offset
                        group_starts.append(start_idx)

            sample_count = len(group_starts) * self._nwin_per_group

            # Skip patients with insufficient files to form a single concatenated group.
            if sample_count > 0:
                self.patient_ranges.append(
                    _PatientFileRange(
                        pid=pid,
                        file_start=file_start,
                        file_count=file_count,
                        group_starts=tuple(group_starts),
                        sample_start=sample_cursor,
                        sample_count=sample_count,
                    )
                )
                sample_cursor += sample_count
            i = j

        if not self.patient_ranges:
            raise ValueError(
                "No patients had enough files to form windows. "
                f"Need at least concat_files={self.concat_files} files per patient."
            )

        # Keep patient_ids consistent with what is actually represented.
        self.patient_ids = [r.pid for r in self.patient_ranges]

        # indices for bisect: start sample index per patient
        self.patient_start_indices = [r.sample_start for r in self.patient_ranges]
        self.total_samples = sample_cursor

        # cache
        self._last_path: Optional[Path] = None
        self._last_x: Optional[torch.Tensor] = None

        self._last_group_key: Optional[tuple[int, int]] = None  # (patient_range_idx, group_start_file_offset)
        self._last_group_x: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        import bisect

        tries = 0
        cur_idx = int(idx)
        while True:
            # map global idx -> patient
            patient_i = bisect.bisect_right(self.patient_start_indices, cur_idx) - 1
            if patient_i < 0:
                patient_i = 0
            pr = self.patient_ranges[patient_i]
            local = cur_idx - pr.sample_start

            group_offset = local // self._nwin_per_group
            win_offset = local - group_offset * self._nwin_per_group

            if group_offset < 0 or group_offset >= len(pr.group_starts):
                # should not happen, but guard
                group_offset = group_offset % max(1, len(pr.group_starts))

            start = win_offset * self._native_stride
            stop = start + self._native_window_size

            try:
                group_start_idx = pr.group_starts[int(group_offset)]
                x = self._load_group(patient_i, group_start_idx)
                if x.shape[1] < stop:
                    raise ValueError(
                        f"Short sequence: need T>={stop} but got T={x.shape[1]} (concat_files={self.concat_files})"
                    )

                xw = x[:, start:stop]
                if self.downsample_factor != 1:
                    xw = xw[:, :: self.downsample_factor]

                if self.axes == 1 and xw.shape[0] != 1:
                    xw = self._reduce_axes(xw)

                xw = xw.float()
                if self.return_id:
                    # Return patient index (not file index) so MELP can form positives across windows.
                    return xw, torch.tensor(patient_i, dtype=torch.long)
                return xw
            except Exception:
                if not self.tolerate_bad_files:
                    raise
                tries += 1
                if tries > self.bad_file_max_retries:
                    raise RuntimeError(
                        f"Exceeded bad_file_max_retries={self.bad_file_max_retries} while fetching idx={idx}."
                    )
                cur_idx = (cur_idx * 1315423911 + 2654435761) % len(self)

    def _load_group(self, patient_range_idx: int, group_start_idx: int) -> torch.Tensor:
        """Load concat_files consecutive files starting at group_start_idx and concat along time."""

        if self.concat_files == 1:
            path = self.file_paths[group_start_idx]
            x = self._load_file(path)
            return self._ensure_channel_first(x)

        key = (int(patient_range_idx), int(group_start_idx))
        if (
            self.cache_last_file
            and self._last_group_key == key
            and self._last_group_x is not None
        ):
            return self._last_group_x

        end = int(group_start_idx) + int(self.concat_files)
        pr = self.patient_ranges[int(patient_range_idx)]
        if group_start_idx < pr.file_start or end > (pr.file_start + pr.file_count):
            raise IndexError(
                "Group would cross patient boundary: "
                f"pid={pr.pid} group_start_idx={group_start_idx} concat_files={self.concat_files} "
                f"patient_file_range=[{pr.file_start}, {pr.file_start + pr.file_count})"
            )
        if end > len(self.file_paths):
            raise IndexError(
                f"group_start_idx={group_start_idx} out of range for concat_files={self.concat_files} with {len(self.file_paths)} files"
            )

        xs = []
        for path in self.file_paths[int(group_start_idx):end]:
            x = self._load_file(path)
            x = self._ensure_channel_first(x)
            xs.append(x)
        x_cat = torch.cat(xs, dim=1)  # [C, T_total]

        if self.cache_last_file:
            self._last_group_key = key
            self._last_group_x = x_cat
        return x_cat

    def _load_file(self, path: Path) -> torch.Tensor:
        if self.cache_last_file and self._last_path == path and self._last_x is not None:
            return self._last_x

        if path.suffix == ".npy":
            arr = np.load(path)
            x = torch.from_numpy(arr)
        elif path.suffix == ".pt":
            x = torch.load(path)
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(
                path,
                columns=self.parquet_columns,
                memory_map=True,
                use_threads=False,
            )
            columns = [
                table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
                for name in self.parquet_columns
            ]
            arr = np.column_stack(columns).astype(np.float32, copy=False)
            x = torch.from_numpy(arr)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        if self.cache_last_file:
            self._last_path = path
            self._last_x = x
        return x

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")

        if self.input_format == "CT":
            if x.shape[0] in (1, 2, 3, 6) and x.shape[1] > x.shape[0]:
                return x
            return x.t()
        else:
            if x.shape[1] in (1, 2, 3, 6) and x.shape[0] > x.shape[1]:
                return x.t()
            return x

    def _reduce_axes(self, x_ct: torch.Tensor) -> torch.Tensor:
        if self.axis_reduce == "l2":
            return torch.sqrt((x_ct**2).sum(dim=0, keepdim=True) + 1e-12)
        if self.axis_reduce == "mean":
            return x_ct.mean(dim=0, keepdim=True)
        if self.axis_reduce == "sum":
            return x_ct.sum(dim=0, keepdim=True)
        raise ValueError(f"Unknown axis_reduce: {self.axis_reduce}")


class DistributedPatientShuffleSampler(Sampler[int]):
    """DDP-safe sampler that shuffles patients (not windows) and yields contiguous indices per patient.

    Assignment of patients to ranks is deterministic via stable hashing, so:
      - no overlap between ranks
      - no need to materialize a gigantic list of window indices

    Works even if torch.distributed isn't initialized yet (falls back to single-rank).
    Distributed state is refreshed lazily on each __iter__/__len__ call so that the
    sampler constructed before DDP init (e.g. during Lightning's step-estimation pass)
    will pick up the correct world_size/rank once DDP is actually running.
    """

    def __init__(
        self,
        dataset: PatientGroupedAccelWindowDataset,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        # Will be populated lazily on first use.
        self._num_replicas: int = 0
        self._rank: int = 0
        self.my_patients: list[int] = []
        self._num_samples: int = 0
        self._refresh_dist_state()

    # ------------------------------------------------------------------
    # Distributed state helpers
    # ------------------------------------------------------------------

    def _current_dist_info(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _refresh_dist_state(self) -> None:
        """Recompute patient assignment if the distributed configuration changed.

        Pads all ranks to the same length so DDP collective ops never deadlock.
        """
        num_replicas, rank = self._current_dist_info()
        if num_replicas == self._num_replicas and rank == self._rank and self.my_patients:
            return  # nothing changed

        self._num_replicas = num_replicas
        self._rank = rank

        rank_samples = [0] * num_replicas
        rank_patients: list[list[int]] = [[] for _ in range(num_replicas)]
        for i, pr in enumerate(self.dataset.patient_ranges):
            r = _stable_hash_to_int(pr.pid, seed=self.seed) % num_replicas
            rank_patients[r].append(i)
            rank_samples[r] += pr.sample_count

        self.my_patients = rank_patients[rank]
        self._num_samples = rank_samples[rank]
        self._padded_num_samples = max(rank_samples) if rank_samples else 0

    # ------------------------------------------------------------------

    def __iter__(self):
        self._refresh_dist_state()

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self.my_patients), generator=g).tolist()
            patient_indices = [self.my_patients[i] for i in order]
        else:
            patient_indices = list(self.my_patients)

        indices: list[int] = []
        for patient_i in patient_indices:
            pr = self.dataset.patient_ranges[patient_i]
            start = pr.sample_start
            end = start + pr.sample_count
            indices.extend(range(start, end))

        if self._padded_num_samples > len(indices) and len(indices) > 0:
            deficit = self._padded_num_samples - len(indices)
            for i in range(deficit):
                indices.append(indices[i % len(indices)])

        yield from indices

    def __len__(self) -> int:
        self._refresh_dist_state()
        return self._padded_num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class PatientShuffleSampler(Sampler[int]):
    """Single-process sampler: shuffle patients then iterate all their sample indices."""

    def __init__(self, dataset: PatientGroupedAccelWindowDataset, *, seed: int = 0):
        self.dataset = dataset
        self.seed = int(seed)

    def __iter__(self):
        patient_order = list(range(len(self.dataset.patient_ranges)))
        g = torch.Generator()
        g.manual_seed(self.seed)
        patient_order = torch.randperm(len(patient_order), generator=g).tolist()
        for patient_i in patient_order:
            pr = self.dataset.patient_ranges[patient_i]
            for sample_idx in range(pr.sample_start, pr.sample_start + pr.sample_count):
                yield sample_idx

    def __len__(self) -> int:
        return len(self.dataset)
