from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowIndex:
    file_idx: int
    start: int


class AccelWindowDataset(Dataset):
    """Windowed accelerometer dataset.

    Supports loading from a directory (or comma-separated list of directories) of:
      - .npy arrays
      - .pt tensors
      - .parquet tables with X/Y/Z columns (configurable)

    Returns *windows* of shape [C, T] (channel-first). The DataLoader collate
    stacks them to [B, C, T].

    Windowing semantics:
      - `window_size` is in timesteps after (optional) resampling.
      - `stride` defaults to `window_size` (non-overlapping windows).
      - windows that don't fully fit are dropped.
    """

    def __init__(
        self,
        data_root: str | Path,
        file_extension: str = ".npy",
        input_format: Literal["CT", "TC"] = "CT",
        parquet_columns: Optional[list[str]] = None,
        max_files: Optional[int] = None,
        concat_files: int = 1,
        window_size: int = 1200,
        stride: Optional[int] = None,
        # fixed-duration fast windowing
        hz: float = 20.0,
        native_hz: float = 20.0,
        file_duration_sec: int = 600,
        assume_fixed_duration: bool = True,
        tolerate_bad_files: bool = True,
        bad_file_max_retries: int = 20,
        cache_last_file: bool = True,
        axes: int = 3,
        axis_reduce: Literal["l2", "mean", "sum"] = "l2",
        return_id: bool = False
    ):
        self.file_extension = file_extension
        self.input_format = input_format
        self.parquet_columns = parquet_columns or ["X", "Y", "Z"]
        self.hz = hz
        self.native_hz = native_hz
        self.concat_files = int(concat_files)
        self.window_size = int(window_size)  # target samples per window
        self.stride = int(stride) if stride is not None else int(window_size)  # target samples stride

        if self.concat_files <= 0:
            raise ValueError(f"concat_files must be >= 1, got {self.concat_files}")

        # internal indexing uses native sample counts (files are stored at native_hz)
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
        self.file_duration_sec = int(file_duration_sec)
        self.assume_fixed_duration = bool(assume_fixed_duration)
        self.tolerate_bad_files = bool(tolerate_bad_files)
        self.bad_file_max_retries = int(bad_file_max_retries)
        self.cache_last_file = bool(cache_last_file)
        self.axes = int(axes)
        self.axis_reduce = axis_reduce
        self.return_id = bool(return_id)

        # Computed once; used when assume_fixed_duration=True.
        self._file_len_T = round(self.file_duration_sec * self.native_hz)
        if self._file_len_T <= 0:
            raise ValueError("file_duration_sec * hz must be > 0")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")

        # Resolve file list
        self.file_paths: list[Path] = []
        roots: list[Path]
        data_root_str = str(data_root)
        if "," in data_root_str:
            roots = [Path(p.strip()) for p in data_root_str.split(",")]
        else:
            roots = [Path(data_root_str)]

        for root in roots:
            if not root.exists():
                raise FileNotFoundError(f"data_root does not exist: {root}")
            # Default to recursive search for all file types to support
            # root/<ID>/<many 10-min parquets> layouts.
            files = sorted(root.glob(f"**/*{self.file_extension}"))
            self.file_paths.extend(files)

        if max_files is not None:
            self.file_paths = self.file_paths[: int(max_files)]

        if not self.file_paths:
            raise ValueError(f"No {self.file_extension} files found under {data_root}")

        # Fast fixed-duration windowing: avoid scanning/parquet reads at init.
        # This is critical for large corpora (millions of files).
        self.window_index: list[WindowIndex] = []
        if self.assume_fixed_duration:
            if self.concat_files != 1 and not self.assume_fixed_duration:
                raise ValueError("concat_files > 1 requires assume_fixed_duration=true")

            self._group_len_T = self._file_len_T * self.concat_files
            self._nwin_per_group = self._compute_windows_per_file(self._group_len_T)
            if self._nwin_per_group <= 0:
                raise ValueError(
                    "No windows per file. Check window_size/stride vs file_duration_sec*hz. "
                    f"Got group_len_T={self._group_len_T}, window_size={self.window_size}, stride={self.stride}."
                )
        else:
            if self.concat_files != 1:
                raise ValueError("concat_files > 1 is only supported with assume_fixed_duration=true")
            # Fallback for variable-length data.
            self._build_window_index()

        # Tiny per-worker cache to reuse the last loaded file when consecutive
        # windows come from the same path.
        self._last_path: Optional[Path] = None
        self._last_x: Optional[torch.Tensor] = None

        # Similar cache for concatenated file groups.
        self._last_group_start: Optional[int] = None
        self._last_group_x: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        if self.assume_fixed_duration:
            n_groups = len(self.file_paths) // self.concat_files
            return n_groups * self._nwin_per_group
        return len(self.window_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Retry loop allows us to tolerate occasional bad/short files without
        # killing training.
        tries = 0
        cur_idx = int(idx)
        while True:
            try:
                group_start, start = self._map_index(cur_idx)

                if self.concat_files == 1:
                    path = self.file_paths[group_start]
                    x = self._load_file(path)
                    x = self._ensure_channel_first(x)  # [C, T]
                else:
                    x = self._load_group(group_start)

                stop = start + self._native_window_size
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
                    return xw, torch.tensor(group_start, dtype=torch.long)
                return xw
            except Exception:
                if not self.tolerate_bad_files:
                    raise
                tries += 1
                if tries > self.bad_file_max_retries:
                    raise RuntimeError(
                        f"Exceeded bad_file_max_retries={self.bad_file_max_retries} while fetching idx={idx}. "
                        "If your data violate the fixed-duration assumption, set assume_fixed_duration=false "
                        "or increase file_duration_sec."
                    )
                # Re-roll a different index (deterministic-ish without extra RNG state):
                cur_idx = (cur_idx * 1315423911 + 2654435761) % len(self)

    # -----------------
    # Internals
    # -----------------

    def _build_window_index(self) -> None:
        for file_idx, path in enumerate(self.file_paths):
            x = self._load_file(path)
            x = self._ensure_channel_first(x)
            T = x.shape[1]
            # Drop incomplete window tail.
            last_start = T - self._native_window_size
            if last_start < 0:
                continue
            for start in range(0, last_start + 1, self.stride):
                self.window_index.append(WindowIndex(file_idx=file_idx, start=start))

        if not self.window_index:
            raise ValueError(
                "No windows could be formed. Check window_size/stride vs file lengths."
            )

    def _compute_windows_per_file(self, T_file: int) -> int:
        last_start = T_file - self._native_window_size
        if last_start < 0:
            return 0
        return 1 + (last_start // self._native_stride)

    def _map_index(self, idx: int) -> tuple[int, int]:
        """Map global idx -> (group_start_file_idx, start)."""
        if self.assume_fixed_duration:
            nwin = self._nwin_per_group
            group_idx = idx // nwin
            win_idx = idx - group_idx * nwin
            start = win_idx * self._native_stride
            group_start = int(group_idx) * int(self.concat_files)
            return int(group_start), int(start)
        wi = self.window_index[idx]
        return wi.file_idx, wi.start

    def _load_group(self, group_start: int) -> torch.Tensor:
        if self.concat_files == 1:
            return self._load_file(self.file_paths[group_start])

        if (
            self.cache_last_file
            and self._last_group_start == group_start
            and self._last_group_x is not None
        ):
            return self._last_group_x

        start = int(group_start)
        end = start + int(self.concat_files)
        if end > len(self.file_paths):
            raise IndexError(
                f"group_start={group_start} out of range for concat_files={self.concat_files} with {len(self.file_paths)} files"
            )

        xs = []
        for path in self.file_paths[start:end]:
            x = self._load_file(path)
            x = self._ensure_channel_first(x)
            xs.append(x)
        x_cat = torch.cat(xs, dim=1)  # [C, T_total]

        if self.cache_last_file:
            self._last_group_start = group_start
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
            import pandas as pd

            df = pd.read_parquet(path, columns=self.parquet_columns)
            arr = df.to_numpy(dtype=np.float32)
            x = torch.from_numpy(arr)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        if self.cache_last_file:
            self._last_path = path
            self._last_x = x
        return x

    def _ensure_channel_first(self, x: torch.Tensor) -> torch.Tensor:
        # parquet loads as [T, C]
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")

        if self.input_format == "CT":
            # already [C, T] for .npy/.pt in your repo; parquet is [T, C]
            if x.shape[0] in (1, 2, 3, 6) and x.shape[1] > x.shape[0]:
                return x
            return x.t()
        else:
            # expected [T, C]
            if x.shape[1] in (1, 2, 3, 6) and x.shape[0] > x.shape[1]:
                return x.t()
            return x

    def _reduce_axes(self, x_ct: torch.Tensor) -> torch.Tensor:
        """Reduce [C, T] -> [1, T]."""
        if self.axis_reduce == "l2":
            # sqrt(x^2 + y^2 + z^2)
            return torch.sqrt((x_ct ** 2).sum(dim=0, keepdim=True) + 1e-12)
        if self.axis_reduce == "mean":
            return x_ct.mean(dim=0, keepdim=True)
        if self.axis_reduce == "sum":
            return x_ct.sum(dim=0, keepdim=True)
        raise ValueError(f"Unknown axis_reduce: {self.axis_reduce}")
