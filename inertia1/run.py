
"""
Unified launcher for time-series self-supervised pretraining in this repo.

Design goals:
- single entrypoint script for the shipped methods (ar_transformer, patchtst)
- single YAML config with CLI overrides (dotlist)
- pretraining axes (window length, sampling rate, uniaxial vs triaxial) are
  controlled entirely through `data.*` overrides (see inertia1/config/default.yaml)

This module intentionally depends only on standard library + PyYAML + torch + (pytorch_)lightning.
"""

from __future__ import annotations

import argparse
import copy
import torch
from pathlib import Path
from typing import Any, Dict

import re
import yaml

# IMPORTANT: Use pytorch_lightning everywhere in this repo.
# Mixing `lightning.pytorch` and `pytorch_lightning` objects can cause
# cryptic runtime errors (notably around `Trainer.fit`).
try:
    import pytorch_lightning as pl
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pytorch_lightning is required. Install it in your environment (e.g., `pip install pytorch-lightning`)."
    ) from e

from inertia1.methods.registry import build_experiment
from inertia1.utils.params import count_parameters


def _pick_ckpt_from_callbacks(trainer: pl.Trainer, pick: str = "best") -> str | None:
    """
    Try to retrieve the checkpoint path produced by ModelCheckpoint.
    """
    ckpt_cbs = [cb for cb in trainer.callbacks if cb.__class__.__name__ == "ModelCheckpoint"]
    if not ckpt_cbs:
        return None
    cb = ckpt_cbs[0]
    if pick == "last":
        p = getattr(cb, "last_model_path", None)
        return p if p else None
    # default best
    p = getattr(cb, "best_model_path", None)
    return p if p else None


def _resolve_ckpt_path(cfg: Dict[str, Any], trainer: pl.Trainer | None = None) -> str | None:
    """
    Resolve which checkpoint file to use for finetune/test.
    Priority:
      1) cfg.run.ckpt_path (if set)
      2) if trainer provided and stage=both: pick from callbacks (best/last)
    """
    run_cfg = cfg.get("run", {}) or {}
    explicit = run_cfg.get("ckpt_path", None)
    if explicit:
        return str(explicit)
    if trainer is not None:
        pick = str(run_cfg.get("ckpt_pick", "best"))
        return _pick_ckpt_from_callbacks(trainer, pick=pick)
    return None


def _resolve_produced_ckpt_path(cfg: Dict[str, Any], trainer: pl.Trainer) -> str | None:
    """Resolve the checkpoint produced by the current trainer run.

    Unlike `_resolve_ckpt_path`, this intentionally ignores `run.ckpt_path`,
    which may point at a prior checkpoint used for finetuning.
    """
    run_cfg = cfg.get("run", {}) or {}
    pick = str(run_cfg.get("ckpt_pick", "best"))
    return _pick_ckpt_from_callbacks(trainer, pick=pick)


def _resolve_pretrain_resume_ckpt_path(cfg: Dict[str, Any]) -> str | None:
    """Resolve optional checkpoint for resuming pretraining."""
    run_cfg = cfg.get("run", {}) or {}
    resume_path = run_cfg.get("resume_ckpt_path", None)
    if resume_path is None:
        return None
    return str(resume_path)


def _warn_if_ckpt_missing(path: str | None):
    if not path:
        return
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d

def _parse_scalar(x: str) -> Any:
    # Try bool/int/float/null; fall back to string
    xl = x.lower()
    if xl in ("true", "false"):
        return xl == "true"
    if xl in ("null", "none"):
        return None
    # int
    try:
        if re.match(r"^[+-]?\d+$", x):
            return int(x)
    except Exception:
        pass
    # float
    try:
        if re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", x):
            return float(x)
    except Exception:
        pass
    return x

def _apply_dotlist(cfg: Dict[str, Any], dotlist: list[str]) -> Dict[str, Any]:
    for item in dotlist:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, raw = item.split("=", 1)
        val: Any
        if raw.startswith("[") and raw.endswith("]"):
            # simple list parser: [a,b,1,true]
            inner = raw[1:-1].strip()
            if not inner:
                val = []
            else:
                parts = [p.strip() for p in inner.split(",")]
                val = [_parse_scalar(p) for p in parts]
        else:
            val = _parse_scalar(raw)

        path = key.split(".")
        cur = cfg
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = val
    return cfg

def load_config(path: str | Path, overrides: list[str] | None = None) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    cfg = dict(cfg)
    if overrides:
        cfg = _apply_dotlist(cfg, overrides)
    return cfg

def load_configs(paths: list[str], overrides: list[str] | None = None) -> Dict[str, Any]:
    """Load and merge multiple YAML configs (later overrides earlier)."""
    cfg: Dict[str, Any] = {}
    for p in paths:
        c = load_config(p, overrides=None)
        cfg = _deep_update(cfg, c)
    if overrides:
        cfg = _apply_dotlist(cfg, overrides)
    return cfg

def infer_preset_name_from_paths(config_paths: list[str]) -> str | None:
    """
    If user passed a config under .../config/presets/<NAME>.yaml,
    return <NAME>.
    """
    for p in config_paths:
        pp = Path(p)
        parts = [x.lower() for x in pp.parts]
        if "presets" in parts:
            return pp.stem  # e.g. medium.yaml -> "medium"
    return None

def apply_preset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a size preset (if present) in a method-aware way.

    Presets live under cfg['preset'] and are namespaced by model family.
    Both shipped methods share the transformer-sizing knobs under
    ``preset.patchtst`` (d_model / n_layers / n_heads / d_ff):
      - patchtst       -> merged into cfg['model']['params'] (backbone)
      - ar_transformer -> merged into cfg['method']['params']
    """
    preset = cfg.get("preset", None)
    if not isinstance(preset, dict):
        return cfg

    method = (cfg.get("method", {}) or {}).get("name", None)
    if method is None:
        return cfg

    cfg = copy.deepcopy(cfg)

    # Ensure containers exist
    cfg.setdefault("method", {}).setdefault("params", {})
    cfg.setdefault("model", {}).setdefault("params", {})

    def merge_into(path_keys: list[str], patch: dict):
        cur = cfg
        for k in path_keys[:-1]:
            cur = cur.setdefault(k, {})
        leaf = cur.setdefault(path_keys[-1], {})
        _deep_update(leaf, patch)

    transformer_knobs = preset.get("patchtst", {})
    if isinstance(transformer_knobs, dict) and transformer_knobs:
        if method == "patchtst":
            merge_into(["model", "params"], transformer_knobs)
        elif method == "ar_transformer":
            merge_into(["method", "params"], transformer_knobs)

    return cfg



def main() -> None:
    parser = argparse.ArgumentParser("inertia1.run", add_help=True)
    parser.add_argument("--config", action="append", default=[str(Path(__file__).parent / "config" / "default.yaml")],
                        help="Path(s) to YAML config. Can be provided multiple times; later files override earlier ones.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Construct the experiment, print resolved config + param counts, then exit.")
    parser.add_argument("overrides", nargs="*", help="Dotlist overrides like data.hz=10 trainer.max_epochs=50")
    args = parser.parse_args()

    cfg = load_configs(args.config, args.overrides)

    preset_name = infer_preset_name_from_paths(args.config)
    if preset_name is not None:
        cfg["preset_name"] = preset_name

    cfg = apply_preset(cfg)

    # Resolve sensor selection into explicit columns/channels early.
    # This ensures modules are initialized with the correct input channel count.
    from inertia1.data.sensor_config import resolve_data_columns_and_channels

    cfg = resolve_data_columns_and_channels(cfg)

    exp = build_experiment(cfg)

    # Print param counts early
    n_total, n_trainable, breakdown = count_parameters(exp["module"], cfg)
    print(f"[inertia1] Params: total={n_total:,} trainable={n_trainable:,}")
    for name, n in breakdown.items():
        print(f"  - {name}: {n:,}")

    fairness = cfg.get("fairness", {})
    max_total = fairness.get("max_total_params", None)
    max_trainable = fairness.get("max_trainable_params", None)
    if max_total is not None and n_total > int(max_total):
        raise ValueError(f"[inertia1] Fairness check failed: total params {n_total:,} > max_total_params {int(max_total):,}")
    if max_trainable is not None and n_trainable > int(max_trainable):
        raise ValueError(f"[inertia1] Fairness check failed: trainable params {n_trainable:,} > max_trainable_params {int(max_trainable):,}")


    if args.dry_run or cfg.get("run", {}).get("dry_run", False):
        print("[inertia1] Dry run enabled. Resolved config:\n")
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    print("[inertia1] Starting training...")

    trainer: pl.Trainer = exp["trainer"]
    module: pl.LightningModule = exp["module"]
    datamodule = exp.get("datamodule")


    run_cfg = cfg.get("run", {}) or {}
    stage = str(run_cfg.get("stage", "pretrain")).lower()
    if stage not in {"pretrain", "finetune", "both"}:
        raise ValueError(f"Invalid run.stage={stage}. Must be pretrain|finetune|both.")

    # --------------------
    # Stage: PRETRAIN
    # --------------------
    pretrain_trainer = trainer
    pretrain_ckpt = None
    if stage in {"pretrain", "both"}:
        print(f"[inertia1] Stage: {stage} -> running pretraining", flush=True)
        try:
            datamodule.setup("fit")
            train_n = getattr(datamodule, "train_dataset", None)
            val_n = getattr(datamodule, "val_dataset", None)
            if train_n is not None:
                print(f"[inertia1] Train samples: {len(train_n):,}")
            if val_n is not None:
                print(f"[inertia1] Val samples:   {len(val_n):,}")
        except Exception as e:
            print(f"[inertia1] Warning: datamodule.setup('fit') failed: {e}")
        resume_ckpt_path = _resolve_pretrain_resume_ckpt_path(cfg)
        if resume_ckpt_path:
            _warn_if_ckpt_missing(resume_ckpt_path)
            print(f"[inertia1] Pretraining resume enabled: ckpt_path={resume_ckpt_path}", flush=True)

        pretrain_trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt_path)
        pretrain_ckpt = _resolve_produced_ckpt_path(cfg, trainer=pretrain_trainer)
        if pretrain_ckpt:
            print(f"[inertia1] Pretraining produced ckpt: {pretrain_ckpt}", flush=True)
        else:
            print("[inertia1] WARNING: Could not auto-resolve a pretraining checkpoint.", flush=True)

    # --------------------
    # Stage: FINETUNE/TEST (MotionFM linear probe)
    # --------------------
    if stage in {"finetune", "both"}:
        try:
            from inertia1.eval.motion_linear_probe import run_motion_linear_probe
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Motion linear probe requires scikit-learn. Install it with `python -m pip install scikit-learn` "
                "in the same environment, or run with `run.stage=pretrain` to skip evaluation."
            ) from e

        ckpt_path = _resolve_ckpt_path(cfg, trainer=pretrain_trainer if stage == "both" else None)
        _warn_if_ckpt_missing(ckpt_path)

        if not ckpt_path:
            raise ValueError(
                "Finetune stage requires a checkpoint. Set run.ckpt_path, or run stage=both "
                "with checkpointing enabled (best/last)."
            )

        print(f"[inertia1] Stage: {stage} -> running finetune/test using ckpt={ckpt_path}", flush=True)

        # Load module weights from checkpoint for embedding extraction
        # NOTE: user is responsible for matching cfg.method/model to the ckpt.
        # We keep strict=False to allow minor key differences; you can tighten later.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[inertia1] WARNING: load_state_dict missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        # Run MotionFM linear probe (train head + test). This is a separate step.
        motion_res = run_motion_linear_probe(cfg, module, ckpt_path)
        print("[inertia1] MotionFM linear-probe done.", flush=True)

        print("[inertia1] MotionFM linear-probe summary (showing first 5):", flush=True)
        lp = (motion_res.get("linear_probe", {}) or {})
        for i, (k, v) in enumerate(lp.items()):
            if i >= 5: break
            print(f"  - {k}: acc={v.get('acc')}", flush=True)

    print("[inertia1] Done.")


if __name__ == "__main__":
    main()
