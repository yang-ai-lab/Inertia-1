
from __future__ import annotations

import os
from typing import Any, Dict

from datetime import datetime

# IMPORTANT: Use pytorch_lightning everywhere in this repo.
# Do NOT mix with `lightning.pytorch` (it can cause runtime issues).
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from pytorch_lightning.loggers import WandbLogger
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pytorch_lightning is required. Install it in your environment (e.g., `pip install pytorch-lightning`)."
    ) from e


def build_trainer(cfg: Dict[str, Any]) -> pl.Trainer:
    tcfg = cfg.get("trainer", {})
    callbacks = []

    validation_disabled = tcfg.get("limit_val_batches", None) in (0, 0.0)

    # Progress bar: force-enabled by default so users can see training is running.
    # Some cluster environments/launchers disable it implicitly; we make it explicit.
    if tcfg.get("enable_progress_bar", True):
        callbacks.append(TQDMProgressBar(refresh_rate=tcfg.get("progress_bar_refresh_rate", 20)))
    if tcfg.get("checkpointing", True):
        ckpt_cfg = cfg.get("checkpointing", {})
        root_dir = ckpt_cfg.get("root_dir", "outputs")
        run_name = ckpt_cfg.get("run_name") or infer_run_name(cfg)
        run_dir = ckpt_cfg.get("run_dir") or os.path.join(root_dir, run_name)
        
        ckpt_cfg["run_name"] = run_name
        ckpt_cfg["run_dir"] = run_dir

        # subdirs
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        log_dir  = os.path.join(run_dir, "logs")

        # Create the directories up-front so users can immediately see where outputs go.
        # (Lightning will also create them on first write.)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        every_n_train_steps = ckpt_cfg.get("every_n_train_steps", None)

        filename = ckpt_cfg.get("filename", "epoch={epoch:04d}-step={step}")
        # When checkpointing during training, avoid templates requiring val metrics.
        if every_n_train_steps not in (None, 0) and isinstance(filename, str) and "val" in filename:
            filename = "epoch={epoch:04d}-step={step}"

        # (1) Periodic step checkpoints: keep intermediates regardless of monitored metrics.
        # This is what you want for "save every 1000 steps" and keep them all.
        if every_n_train_steps not in (None, 0):
            callbacks.append(
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename=filename,
                    save_top_k=-1,  # keep all step checkpoints
                    save_last=bool(ckpt_cfg.get("save_last", True)),
                    every_n_train_steps=every_n_train_steps,
                    every_n_epochs=None,
                    train_time_interval=ckpt_cfg.get("train_time_interval", None),
                    save_on_train_epoch_end=bool(ckpt_cfg.get("save_on_train_epoch_end", False)),
                )
            )

        # (2) Best checkpoint: separate callback so "best" selection doesn't delete intermediates.
        if not validation_disabled:
            monitor = ckpt_cfg.get("monitor", "val/loss")
            mode = ckpt_cfg.get("mode", "min")
            if cfg.get("data", {}).get("val_split", 0.1) in (0, 0.0, None):
                monitor = "train/loss"
                mode = "min"

            callbacks.append(
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    monitor=monitor,
                    mode=mode,
                    filename=filename,
                    save_top_k=int(ckpt_cfg.get("save_top_k", 1)),
                    save_last=False,  # handled by the periodic callback when enabled
                    every_n_epochs=ckpt_cfg.get("every_n_epochs", 1),
                    every_n_train_steps=None,
                    train_time_interval=None,
                    save_on_train_epoch_end=bool(ckpt_cfg.get("save_on_train_epoch_end", False)),
                )
            )

    if tcfg.get("enable_lr_monitor", True):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    logger = None
    log_cfg = cfg.get("logging", {})
    if log_cfg.get("wandb", {}).get("enabled", False):
        wandb_cfg = log_cfg.get("wandb", {})
        run_name = cfg["checkpointing"]["run_name"]
        run_dir  = cfg["checkpointing"]["run_dir"]

        # Only fill defaults if user didn't specify them
        wandb_name = wandb_cfg.get("name") or run_name
        wandb_save_dir = wandb_cfg.get("save_dir") or run_dir

        # if you're using tags, optionally add preset/method tags automatically
        tags = wandb_cfg.get("tags", None)
        if tags is None:
            tags = []
        if isinstance(tags, (list, tuple)):
            method = (cfg.get("method", {}) or {}).get("name", None)
            preset = cfg.get("preset_name", None)
            if method and method not in tags:
                tags.append(method)
            if preset and preset not in tags:
                tags.append(preset)

        logger = WandbLogger(
            project=wandb_cfg.get("project", "inertia1"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_name,
            save_dir=wandb_save_dir,
            tags=tags,
            # optionally: config=cfg  (but can be huge)
        )

    # Map common trainer fields only (avoid passing unknown keys)
    trainer_kwargs = dict(
        accelerator=tcfg.get("accelerator", "auto"),
        devices=tcfg.get("devices", "auto"),
        strategy=tcfg.get("strategy", "auto"),
        precision=tcfg.get("precision", "32-true"),
        max_epochs=tcfg.get("max_epochs", 100),
        log_every_n_steps=tcfg.get("log_every_n_steps", 50),
        enable_progress_bar=tcfg.get("enable_progress_bar", True),
        enable_model_summary=tcfg.get("enable_model_summary", True),
        enable_checkpointing=tcfg.get("enable_checkpointing", True),
        callbacks=callbacks,
        logger=logger,
    )

    # Gradient clipping (critical for stable training of larger models)
    if "gradient_clip_val" in tcfg:
        trainer_kwargs["gradient_clip_val"] = tcfg["gradient_clip_val"]
    if "gradient_clip_algorithm" in tcfg:
        trainer_kwargs["gradient_clip_algorithm"] = tcfg["gradient_clip_algorithm"]

    # Common Lightning limit knobs (needed for config/dotlist overrides)
    for key in (
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches",
    ):
        if key in tcfg:
            trainer_kwargs[key] = tcfg[key]

    # Validation cadence:
    # - By default Lightning validates at the end of each epoch.
    # - If we're saving periodic step checkpoints, it is often useful to validate on the
    #   same cadence so `val/loss` (and "best") updates regularly.
    # Users can always override by setting trainer.val_check_interval in the config.
    if validation_disabled:
        # Fully disable validation (avoid long/buggy val loops). Periodic step checkpoints still work.
        trainer_kwargs.pop("val_check_interval", None)
        trainer_kwargs.setdefault("check_val_every_n_epoch", 10**9)
        trainer_kwargs.setdefault("num_sanity_val_steps", 0)
    else:
        if "val_check_interval" in tcfg:
            trainer_kwargs["val_check_interval"] = tcfg["val_check_interval"]
        else:
            ckpt_cfg = cfg.get("checkpointing", {})
            every_n_train_steps = ckpt_cfg.get("every_n_train_steps", None)
            if every_n_train_steps not in (None, 0):
                trainer_kwargs["val_check_interval"] = int(every_n_train_steps)

    if "check_val_every_n_epoch" in tcfg:
        trainer_kwargs["check_val_every_n_epoch"] = tcfg["check_val_every_n_epoch"]
    if "num_sanity_val_steps" in tcfg:
        trainer_kwargs["num_sanity_val_steps"] = tcfg["num_sanity_val_steps"]

    # IMPORTANT: Prevent Lightning from automatically wrapping/replacing the DataLoader sampler
    # (e.g., with DistributedSampler), which can materialize huge permutations and OOM on large datasets.
    # This is needed when using custom patient-level samplers.
    import inspect

    sig = inspect.signature(pl.Trainer)
    if "use_distributed_sampler" in sig.parameters:
        trainer_kwargs["use_distributed_sampler"] = bool(tcfg.get("use_distributed_sampler", True))
    elif "replace_sampler_ddp" in sig.parameters:
        # Older Lightning versions used replace_sampler_ddp with inverse semantics.
        use_ds = bool(tcfg.get("use_distributed_sampler", True))
        trainer_kwargs["replace_sampler_ddp"] = use_ds
    # Optional deterministic
    if "deterministic" in tcfg:
        trainer_kwargs["deterministic"] = tcfg["deterministic"]
    return pl.Trainer(**trainer_kwargs)


def build_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve sensor selection (if configured) into parquet_columns/channels
    # before constructing modules that need a fixed input channel count.
    try:
        from inertia1.data.sensor_config import resolve_data_columns_and_channels

        cfg = resolve_data_columns_and_channels(cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to resolve data columns/channels: {e}")

    method = cfg.get("method", {}).get("name", None)
    if not method:
        raise ValueError("Config must set method.name (e.g., ar_transformer, patchtst).")

    # Import adapters lazily to avoid heavy imports.
    # This open-source release ships two representative pretraining methods:
    #   - ar_transformer: patch-level autoregressive transformer
    #   - patchtst:       masked patch reconstruction (PatchTST)
    if method == "ar_transformer":
        from inertia1.methods.ar_transformer import build_ar_transformer_experiment
        return build_ar_transformer_experiment(cfg, build_trainer(cfg))
    elif method == "patchtst":
        from inertia1.methods.patchtst import build_patchtst_experiment
        return build_patchtst_experiment(cfg, build_trainer(cfg))
    else:
        raise ValueError(
            f"Unknown method.name='{method}'. Expected one of: ar_transformer, patchtst."
        )
    
def infer_run_name(cfg: dict) -> str:
    method = cfg.get("method", {}).get("name", "unknown")

    preset_name = cfg.get("preset_name", None)

    d = cfg.get("data", {})
    win = d.get("window_sec")
    hz = d.get("hz")
    axes = d.get("axes")
    seed = d.get("seed", None)

    mp = cfg.get("model", {}).get("params", {})
    p = cfg.get("method", {}).get("params", {})
    tag = None

    if method == "patchtst":
        tag = f"dm{mp.get('d_model')}_L{mp.get('n_layers')}_H{mp.get('n_heads')}"
    elif method == "ar_transformer":
        tag = f"dm{p.get('d_model')}_L{p.get('n_layers')}_H{p.get('n_heads')}"

    parts = [method]
    if preset_name: parts.append(preset_name)
    if tag: parts.append(str(tag))
    if win is not None: parts.append(f"win{win}")
    if hz is not None: parts.append(f"hz{hz}")
    if axes is not None: parts.append(f"ax{axes}")
    if seed is not None: parts.append(f"seed{seed}")

    name = "__".join(parts)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}__{timestamp}"
