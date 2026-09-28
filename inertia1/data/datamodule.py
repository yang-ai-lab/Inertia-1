from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, random_split

try:
    import pytorch_lightning as pl
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pytorch_lightning is required. Install it in your environment (e.g., `pip install pytorch-lightning`)."
    ) from e

from .accel_dataset import AccelWindowDataset
from .patient_dataset import (
    DistributedPatientShuffleSampler,
    PatientGroupedAccelWindowDataset,
    PatientShuffleSampler,
)


class UnifiedDataModule(pl.LightningDataModule):
    """Shared DataModule used by all methods.

    This is intentionally *minimal* at first, focusing on:
      - consistent shapes ([B, C, T])
      - windowing + stride
      - axis reduction (3->1)

    Resampling and time/freq representations will be layered in next.
    """

    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path | None = None,
        batch_size: int = 64,
        val_split: float = 0.1,
        train_subsample_ratio: float = 1.0,
        val_subsample_ratio: float = 0.2,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        file_extension: str = ".parquet",
        input_format: str = "CT",
        parquet_columns: Optional[list[str]] = None,
        max_files: Optional[int] = None,
        # windowing
        window_sec: Optional[float] = None,
        hz: float = 20.0,
        native_hz: float = 20.0,
        file_duration_sec: int = 600,
        assume_fixed_duration: bool = True,
        tolerate_bad_files: bool = True,
        bad_file_max_retries: int = 20,
        cache_last_file: bool = True,
        window_size: Optional[int] = None,
        stride_sec: Optional[float] = None,
        stride: Optional[int] = None,
        # axes
        axes: int = 3,
        axis_reduce: str = "l2",
        concat_files: int = 1,
        require_consecutive_files: bool = True,
        return_id: bool = False,
        patient_shuffle: bool = True,
        patient_shuffle_seed: int = 42,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split_csv = Path(split_csv) if split_csv is not None else None
        self.batch_size = int(batch_size)
        self.val_split = float(val_split)
        self.train_subsample_ratio = float(train_subsample_ratio)
        self.val_subsample_ratio = float(val_subsample_ratio)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.seed = int(seed)
        self.file_extension = file_extension
        self.input_format = input_format
        self.parquet_columns = parquet_columns
        self.max_files = max_files
        self.hz = hz
        self.native_hz = native_hz
        self.file_duration_sec = int(file_duration_sec)
        self.assume_fixed_duration = bool(assume_fixed_duration)
        self.tolerate_bad_files = bool(tolerate_bad_files)
        self.bad_file_max_retries = int(bad_file_max_retries)
        self.cache_last_file = bool(cache_last_file)
        self.patient_shuffle = bool(patient_shuffle)
        self.patient_shuffle_seed = int(patient_shuffle_seed)

        # Resolve window_size/stride in timesteps.
        if window_size is None:
            if window_sec is None:
                raise ValueError("Provide either window_size (timesteps) or window_sec + hz")
            window_size = int(round(float(window_sec) * self.hz))
        if stride is None:
            if stride_sec is not None:
                stride = int(round(float(stride_sec) * self.hz))
            else:
                stride = int(window_size)

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.axes = int(axes)
        self.axis_reduce = axis_reduce
        self.concat_files = int(concat_files)
        self.require_consecutive_files = bool(require_consecutive_files)
        self.return_id = bool(return_id)

        self.train_dataset = None
        self.val_dataset = None
        self._train_sampler = None
        self._val_sampler = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            # Datasets already built; samplers will refresh their DDP state lazily.
            return

        # Patient-level split + sampling (avoids materializing huge window index lists).
        # Enabled when split_csv is provided (expects patient_id,split with train/val).
        if self.split_csv is not None:
            train_ds = PatientGroupedAccelWindowDataset(
                data_root=self.data_root,
                split_csv=self.split_csv,
                split="train",
                file_extension=self.file_extension,
                input_format=self.input_format,  # type: ignore[arg-type]
                parquet_columns=self.parquet_columns,
                max_files=self.max_files,
                window_size=self.window_size,
                stride=self.stride,
                hz=self.hz,
                native_hz=self.native_hz,
                file_duration_sec=self.file_duration_sec,
                assume_fixed_duration=self.assume_fixed_duration,
                tolerate_bad_files=self.tolerate_bad_files,
                bad_file_max_retries=self.bad_file_max_retries,
                cache_last_file=self.cache_last_file,
                axes=self.axes,
                axis_reduce=self.axis_reduce,  # type: ignore[arg-type]
                concat_files=self.concat_files,
                require_consecutive_files=self.require_consecutive_files,
                return_id=self.return_id,
                seed=self.seed,
                patient_subsample_ratio=self.train_subsample_ratio,
            )
            val_ds = PatientGroupedAccelWindowDataset(
                data_root=self.data_root,
                split_csv=self.split_csv,
                split="val",
                file_extension=self.file_extension,
                input_format=self.input_format,  # type: ignore[arg-type]
                parquet_columns=self.parquet_columns,
                max_files=None,
                window_size=self.window_size,
                stride=self.stride,
                hz=self.hz,
                native_hz=self.native_hz,
                file_duration_sec=self.file_duration_sec,
                assume_fixed_duration=self.assume_fixed_duration,
                tolerate_bad_files=self.tolerate_bad_files,
                bad_file_max_retries=self.bad_file_max_retries,
                cache_last_file=self.cache_last_file,
                axes=self.axes,
                axis_reduce=self.axis_reduce,  # type: ignore[arg-type]
                concat_files=self.concat_files,
                require_consecutive_files=self.require_consecutive_files,
                return_id=self.return_id,
                seed=self.seed,
                patient_subsample_ratio=self.val_subsample_ratio,
            )

            self.train_dataset = train_ds
            self.val_dataset = val_ds

            if self.patient_shuffle:
                # DDP-aware; safe in single-process too.
                self._train_sampler = DistributedPatientShuffleSampler(
                    train_ds,
                    shuffle=True,
                    seed=self.patient_shuffle_seed,
                )
            else:
                # Still provide a sampler to avoid Lightning wrapping with a huge DistributedSampler.
                self._train_sampler = DistributedPatientShuffleSampler(
                    train_ds,
                    shuffle=False,
                    seed=self.patient_shuffle_seed,
                )
            self._val_sampler = DistributedPatientShuffleSampler(
                val_ds,
                shuffle=False,
                seed=self.patient_shuffle_seed,
            )
            return

        full = AccelWindowDataset(
            data_root=self.data_root,
            file_extension=self.file_extension,
            input_format=self.input_format,
            parquet_columns=self.parquet_columns,
            max_files=self.max_files,
            concat_files=self.concat_files,
            window_size=self.window_size,
            stride=self.stride,
            hz=self.hz,
            native_hz=self.native_hz,
            file_duration_sec=self.file_duration_sec,
            assume_fixed_duration=self.assume_fixed_duration,
            tolerate_bad_files=self.tolerate_bad_files,
            bad_file_max_retries=self.bad_file_max_retries,
            cache_last_file=self.cache_last_file,
            axes=self.axes,
            axis_reduce=self.axis_reduce,  # type: ignore[arg-type]
        )

        total = len(full)
        val_size = int(total * self.val_split)
        train_size = total - val_size

        g = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(full, [train_size, val_size], generator=g)

        # Optionally subsample the training set (useful for quick experiments).
        if self.train_subsample_ratio < 1.0:
            if not (0.0 < self.train_subsample_ratio <= 1.0):
                raise ValueError(f"train_subsample_ratio must be in (0,1], got {self.train_subsample_ratio}")
            n_keep = max(1, int(round(len(self.train_dataset) * self.train_subsample_ratio)))
            # Deterministic subsample using the same seed as the split.
            idx = torch.randperm(len(self.train_dataset), generator=g)[:n_keep].tolist()
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, idx)

    def train_dataloader(self) -> DataLoader:
        if self._train_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self._train_sampler,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                **self._loader_perf_kwargs(),
                drop_last=True,
                collate_fn=self._collate,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self._loader_perf_kwargs(),
            drop_last=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        val_set = self.val_dataset
        # Only subsample via random Subset when NOT using the patient sampler.
        # When _val_sampler is set (split_csv path), the dataset was already built
        # with patient_subsample_ratio applied, so a second Subset would cause the
        # sampler to yield indices outside the Subset bounds (IndexError).
        if self._val_sampler is None and self.val_subsample_ratio < 1.0:
            n = len(val_set)
            n_keep = max(1, int(round(n * self.val_subsample_ratio)))
            g = torch.Generator().manual_seed(self.seed)
            idx = torch.randperm(n, generator=g)[:n_keep].tolist()
            from torch.utils.data import Subset
            val_set = Subset(val_set, idx)
        if self._val_sampler is not None:
            return DataLoader(
                val_set,
                batch_size=self.batch_size,
                sampler=self._val_sampler,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                **self._loader_perf_kwargs(),
                drop_last=False,
                collate_fn=self._collate,
            )
        return DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self._loader_perf_kwargs(),
            drop_last=False,
            collate_fn=self._collate,
        )

    def _loader_perf_kwargs(self) -> Dict[str, Any]:
        if self.num_workers <= 0:
            return {}
        return {
            "persistent_workers": True,
            "prefetch_factor": 4,
        }

    @staticmethod
    def _collate(batch):
        """Collate windows into a batch.

        Supported dataset item formats:
        - x: Tensor[C, T] -> returns Tensor[B, C, T]
        - (x, id): -> returns (Tensor[B, C, T], Tensor[B])
        """
        if not batch:
            raise ValueError("Empty batch")

        first = batch[0]
        if torch.is_tensor(first):
            # batch: list[Tensor[C, T]] -> Tensor[B, C, T]
            return torch.stack(batch).float()

        if isinstance(first, (tuple, list)) and len(first) == 2:
            xs, ids = zip(*batch)
            x = torch.stack(list(xs)).float()
            id_tensors = []
            for i in ids:
                if torch.is_tensor(i):
                    id_tensors.append(i)
                else:
                    id_tensors.append(torch.tensor(i))
            ids_batch = torch.stack(id_tensors).long()
            return x, ids_batch

        raise TypeError(
            "Unsupported batch element type for collate: "
            f"{type(first)} (example={first!r}). Expected Tensor or (Tensor, id)."
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "split_csv": str(self.split_csv) if self.split_csv is not None else None,
            "file_extension": self.file_extension,
            "hz": self.hz,
            "file_duration_sec": self.file_duration_sec,
            "assume_fixed_duration": self.assume_fixed_duration,
            "window_size": self.window_size,
            "stride": self.stride,
            "axes": self.axes,
            "axis_reduce": self.axis_reduce,
            "batch_size": self.batch_size,
        }
