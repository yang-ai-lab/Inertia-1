from __future__ import annotations

from typing import Any, Dict

from inertia1.data.datamodule import UnifiedDataModule


def build_datamodule(cfg: Dict[str, Any]) -> UnifiedDataModule:
    dcfg = cfg.get("data", {})

    return UnifiedDataModule(
        data_root=dcfg["data_root"],
        split_csv=dcfg.get("split_csv", None),
        batch_size=dcfg.get("batch_size", 64),
        val_split=dcfg.get("val_split", 0.1),
        train_subsample_ratio=dcfg.get("train_subsample_ratio", 1.0),
        val_subsample_ratio=dcfg.get("val_subsample_ratio", 0.2),
        num_workers=dcfg.get("num_workers", 8),
        pin_memory=dcfg.get("pin_memory", True),
        seed=dcfg.get("seed", 42),
        file_extension=dcfg.get("file_extension", ".parquet"),
        input_format=dcfg.get("input_format", "CT"),
        parquet_columns=dcfg.get("parquet_columns", None),
        max_files=dcfg.get("max_files", dcfg.get("max_samples", None)),
        window_sec=dcfg.get("window_sec", 60),
        stride_sec=dcfg.get("stride_sec", None),
        hz=dcfg.get("hz", 20),
        native_hz=dcfg.get("native_hz", 20),
        file_duration_sec=dcfg.get("file_duration_sec", 600),
        assume_fixed_duration=dcfg.get("assume_fixed_duration", True),
        tolerate_bad_files=dcfg.get("tolerate_bad_files", True),
        bad_file_max_retries=dcfg.get("bad_file_max_retries", 20),
        cache_last_file=dcfg.get("cache_last_file", True),
        axes=dcfg.get("axes", 3),
        axis_reduce=dcfg.get("axis_reduce", "l2"),
        concat_files=dcfg.get("concat_files", 1),
        require_consecutive_files=dcfg.get("require_consecutive_files", True),
        return_id=dcfg.get("return_id", False),
        patient_shuffle=dcfg.get("patient_shuffle", True),
        patient_shuffle_seed=dcfg.get("patient_shuffle_seed", dcfg.get("seed", 42)),
    )
