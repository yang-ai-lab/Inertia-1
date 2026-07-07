from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc
import math
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# matplotlib for plots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import time


def _make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    enabled: bool,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
):
    """Return an optional per-step warmup+cosine LR scheduler."""
    if not enabled:
        return None
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(decay_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def _load_motion_module(py_path: str):
    py_path = str(py_path)
    spec = importlib.util.spec_from_file_location("motion_dataloader", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load motion dataloader module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _infer_axial_mode(cfg: Dict[str, Any]) -> str:
    axes = int((cfg.get("data", {}) or {}).get("axes", 3))
    return "uniaxial" if axes == 1 else "triaxial"


def _infer_hz_and_window(cfg: Dict[str, Any]) -> Tuple[float, int]:
    ecfg = (cfg.get("eval_motion", {}) or {})
    dcfg = (cfg.get("data", {}) or {})
    hz = float(ecfg.get("sampling_rate") or dcfg.get("hz", 20))
    window_sec = int(ecfg.get("window_sec") or dcfg.get("window_sec", 60))
    return hz, window_sec


def _get_model_info(module: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model information from module and config."""
    info = {
        "method": cfg.get("method", {}).get("name", "unknown"),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Try to get parameter counts
    try:
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        info["total_params"] = int(total_params)
        info["trainable_params"] = int(trainable_params)
    except:
        pass
    
    # Get config info
    method_cfg = cfg.get("method", {}).get("params", None)
    if method_cfg:
        info["method_config"] = {k: v for k, v in method_cfg.items() if isinstance(v, (int, float, str, bool))}
    
    # Get checkpoint info if available
    ckpt_cfg = cfg.get("checkpointing", {})
    if ckpt_cfg:
        info["checkpoint"] = ckpt_cfg.get("checkpoint_path", "none")
    
    return info


def _stratified_split(labels: np.ndarray, seed: int, train_frac: float, val_frac: float):
    """Stratified split that handles edge cases."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = np.arange(n)

    # Group by class
    train_idx, val_idx, test_idx = [], [], []
    unique_labels = np.unique(labels)
    
    # Check if we have enough samples per class for splitting
    min_samples_per_class = min(np.sum(labels == c) for c in unique_labels)
    if min_samples_per_class < 3:
        # Not enough samples for proper split - do simple random split
        rng.shuffle(idx)
        n_train = max(1, int(train_frac * n))
        n_val = max(0, int(val_frac * n))
        return idx[:n_train].tolist(), idx[n_train:n_train+n_val].tolist(), idx[n_train+n_val:].tolist()
    
    for c in unique_labels:
        c_idx = idx[labels == c]
        rng.shuffle(c_idx)
        n_c = len(c_idx)
        n_train = max(1, int(round(train_frac * n_c)))
        n_val = max(0, int(round(val_frac * n_c)))
        n_train = min(n_train, n_c - 1)
        n_val = min(n_val, n_c - n_train - 1)
        
        train_idx.append(c_idx[:n_train])
        val_idx.append(c_idx[n_train:n_train + n_val])
        test_idx.append(c_idx[n_train + n_val:])
    
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
def _subsample_indices_stratified(
    indices: list[int] | np.ndarray,
    labels: np.ndarray,
    ratio: float,
    seed: int,
) -> list[int]:
    """Deterministically subsample a set of indices while preserving class coverage.

    Picks approximately `ratio` of indices per class from the provided `indices`.
    This is designed to be safe with frozen splits because it *does not* change dataset
    window indexing; it only selects a subset of the split indices.
    """
    if ratio is None:
        return list(indices)

    r = float(ratio)
    if r >= 1.0:
        return list(indices)
    if not (0.0 < r <= 1.0):
        raise ValueError(f"train_subsample_ratio must be in (0,1], got {ratio}")

    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return []

    y = labels[idx]
    rng = np.random.default_rng(int(seed))

    kept: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        if cls_idx.size == 0:
            continue
        rng.shuffle(cls_idx)
        n_keep = max(1, int(round(cls_idx.size * r)))
        kept.append(cls_idx[:n_keep])

    if not kept:
        return []

    out = np.concatenate(kept)
    rng.shuffle(out)
    return out.tolist()


def _cap_indices_per_class_within_split(
    indices: list[int] | np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    seed: int,
) -> list[int]:
    """Randomly cap windows per class within a single split index list.

    Does not change global dataset indexing. If a class appears in this split at
    least once, it still appears at least once after capping (for max_per_class >= 1).
    """
    if max_per_class is None or int(max_per_class) <= 0:
        return list(indices)

    k = int(max_per_class)
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return []

    y = labels[idx]
    rng = np.random.default_rng(int(seed))
    parts: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        if cls_idx.size == 0:
            continue
        rng.shuffle(cls_idx)
        take = min(cls_idx.size, k)
        parts.append(cls_idx[:take])

    if not parts:
        return []

    out = np.concatenate(parts)
    rng.shuffle(out)
    return out.tolist()


def _labels_present(indices: list[int], labels: np.ndarray) -> set[int]:
    if not indices:
        return set()
    idx = np.asarray(indices, dtype=int)
    return set(np.unique(labels[idx]).tolist())


class _EncodeWrapper(nn.Module):
    """Adapter that exposes `backbone.encode(x)` as a regular `forward(x)`."""

    def __init__(self, backbone: Any):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)


def _make_encoder(module: Any, device: torch.device) -> nn.Module:
    """Wrap `module.encode` as a forward()-callable nn.Module on `device`."""
    return _EncodeWrapper(module).to(device)


@torch.no_grad()
def _encode_dataset(module, dl, device):
    module.eval()
    Z_chunks = []
    Y_chunks = []

    if len(dl) == 0:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    encoder = _make_encoder(module, device)

    # Embedding extraction is a single forward pass per split; the speedup from
    # bf16 autocast here is negligible, but precision loss directly degrades
    # downstream LP quality (AR-Transformer attention is bf16-sensitive enough
    # that HHAR-scale signals get clobbered). Keep this in fp32 for fidelity.
    for batch in tqdm(dl, total=len(dl)):
        x, y = batch[0], batch[1]

        if x.dtype != torch.float32:
            x = x.float()

        x = x.to(device, non_blocking=True)

        z = encoder(x)

        Z_chunks.append(z.float())  # keep on GPU
        Y_chunks.append(y if torch.is_tensor(y) else torch.tensor(y))

    if not Z_chunks:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    Z = torch.cat(Z_chunks, dim=0)
    Y = torch.cat([t if torch.is_tensor(t) else torch.tensor(t) for t in Y_chunks], dim=0)

    # one copy at the end
    return Z.detach().cpu(), Y.detach().cpu()


def _compute_class_weights(
    y_train: torch.Tensor, num_classes: int, scheme: str = "inv_freq"
) -> torch.Tensor:
    """Per-class weights for nn.CrossEntropyLoss(weight=...).

    Schemes:
      - "inv_freq":  w_c = N / (K * n_c)   (sklearn 'balanced')
      - "inv_sqrt":  w_c = sqrt(N / n_c)   (gentler reweighting)

    Weights are normalized to mean=1 so the loss magnitude stays comparable to
    the unweighted CE (so existing ES patience/min_delta values still make sense).
    """
    counts = torch.bincount(y_train.long(), minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    n = counts.sum()
    s = (scheme or "inv_freq").lower()
    if s == "inv_freq":
        w = n / (float(num_classes) * counts)
    elif s == "inv_sqrt":
        w = torch.sqrt(n / counts)
    else:
        raise ValueError(f"Unknown class_weight scheme={scheme!r}")
    w = w * (float(num_classes) / w.sum())
    return w


def _fit_linear_head(
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    Z_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    batch_size: int = 256,
    early_stopping: Dict[str, Any] | None = None,
    class_balanced_loss: bool = True,
    class_weight_scheme: str = "inv_freq",
):
    """Fit linear head with optional class-balanced cross-entropy."""
    torch.manual_seed(seed)

    # Keep features in fp32 for linear head training.
    # (Z_* are CPU tensors coming from _encode_dataset)
    if Z_train.dtype != torch.float32:
        Z_train = Z_train.float()

    has_val = (Z_val is not None) and (Z_val.ndim == 2) and (Z_val.shape[0] > 0) and (Z_val.shape[1] == Z_train.shape[1])
    if has_val and (Z_val.dtype != torch.float32):
        Z_val = Z_val.float()

    head = nn.Linear(Z_train.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    cls_weights = None
    if class_balanced_loss:
        cls_weights = _compute_class_weights(y_train, num_classes, scheme=class_weight_scheme).to(device)
        print(
            f"    class_balanced_loss=True scheme={class_weight_scheme} "
            f"weights[min/mean/max]={cls_weights.min().item():.3f}/"
            f"{cls_weights.mean().item():.3f}/{cls_weights.max().item():.3f}"
        )
    loss_fn = nn.CrossEntropyLoss(weight=cls_weights)

    best_val = float("inf")
    best_train = float("inf")
    best_state = None
    best_epoch: int | None = None
    train_history = []
    val_history = []

    es_cfg = early_stopping or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = es_cfg.get("patience", None)
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_warmup = int(es_cfg.get("warmup_epochs", 0))
    # Hard floor: never early-stop before this many epochs have completed,
    # regardless of patience/warmup.
    es_min_epochs = int(es_cfg.get("min_epochs", 0))

    # If patience is not set, treat early stopping as disabled.
    if es_patience is None:
        es_enabled = False
    es_patience = int(es_patience) if es_enabled else 0

    bad_epochs = 0

    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    for epoch in range(int(epochs)):
        head.train()
        perm = torch.randperm(Z_train.shape[0])
        bs = batch_size
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, Z_train.shape[0], bs):
            idx = perm[i:i+bs]
            xb = Z_train[idx].to(device, dtype=head.weight.dtype)
            yb = y_train[idx].to(device).long()
            logits = head(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            del xb, yb, logits, loss

        avg_train_loss = epoch_loss / n_batches
        train_history.append(avg_train_loss)

        head.eval()
        if has_val:
            with torch.no_grad():
                logits = head(Z_val.to(device, dtype=head.weight.dtype))
                vloss = loss_fn(logits, y_val.to(device).long()).item()
                val_history.append(vloss)
                del logits

            if vloss < (best_val - es_min_delta):
                best_val = vloss
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            else:
                bad_epochs += 1
        else:
            # No val split: monitor training loss.
            val_history.append(float("nan"))
            if avg_train_loss < (best_train - es_min_delta):
                best_train = avg_train_loss
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            else:
                bad_epochs += 1

        if (epoch + 1) % 5 == 0:
            if has_val:
                print(f"    Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_history[-1]:.4f}")
            else:
                print(f"    Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}")

        if (
            es_enabled
            and (epoch + 1) > es_warmup
            and (epoch + 1) >= es_min_epochs
            and bad_epochs >= es_patience
        ):
            monitor = "val_loss" if has_val else "train_loss"
            best = best_val if has_val else best_train
            print(
                f"    EarlyStopping: stop at epoch {epoch+1}/{epochs} (monitor={monitor}, best={best:.6f}, "
                f"patience={es_patience}, min_delta={es_min_delta}, "
                f"warmup={es_warmup}, min_epochs={es_min_epochs})",
                flush=True,
            )
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    fit_info: Dict[str, Any] = {
        "monitor": "val_loss" if has_val else "train_loss",
        "best_epoch": (int(best_epoch) + 1) if best_epoch is not None else None,
        "best_val_loss": float(best_val) if has_val else None,
        "best_train_loss": float(best_train) if not has_val else None,
        "epochs_ran": int(len(train_history)),
        "early_stopping": {
            "enabled": bool(es_enabled),
            "patience": int(es_patience) if es_enabled else None,
            "min_delta": float(es_min_delta),
            "warmup_epochs": int(es_warmup),
            "min_epochs": int(es_min_epochs),
        },
    }

    return head, train_history, val_history, fit_info


def _fit_full_finetune(
    module: Any,
    train_dl: DataLoader,
    val_dl: DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    max_grad_norm: float | None = 1.0,
    early_stopping: Dict[str, Any] | None = None,
    class_balanced_loss: bool = True,
    class_weight_scheme: str = "inv_freq",
    train_labels: torch.Tensor | None = None,
    scheduler_cfg: Dict[str, Any] | None = None,
) -> Tuple[nn.Module, List[float], List[float], Dict[str, Any]]:
    """Finetune backbone + linear classifier on raw windows."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ensure backbone is trainable
    module.train()
    for p in module.parameters():
        p.requires_grad = True

    def _build_encode_head(backbone: Any, head: nn.Module, device: torch.device) -> nn.Module:
        class _EncodeHead(nn.Module):
            def __init__(self, bb: Any, hd: nn.Module):
                super().__init__()
                self.backbone = bb
                self.head = hd

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z = self.backbone.encode(x)
                return self.head(z)

        return _EncodeHead(backbone, head).to(device)

    emb_probe = _make_encoder(module, device)
    xb, yb = next(iter(train_dl))
    xb = xb.to(device).float()
    with torch.no_grad():
        zb = emb_probe(xb)
    emb_dim = int(zb.shape[-1])
    del xb, yb, zb, emb_probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    head = nn.Linear(emb_dim, num_classes).to(device)
    params = list(module.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler_cfg = scheduler_cfg or {}
    n_batches_per_epoch = max(1, len(train_dl))
    sched_warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0) or 0)
    # Cosine horizon controls when the cosine decay reaches min_lr_ratio. If
    # unset (<=0), fall back to total epochs. Setting this to a realistic
    # finetuning budget (e.g. ES min_epochs + patience) prevents the previous
    # behavior where cosine barely decayed before early-stopping fired.
    sched_cosine_epochs = int(scheduler_cfg.get("cosine_epochs", 0) or 0)
    if sched_cosine_epochs <= 0:
        sched_cosine_epochs = int(epochs)
    sched_total_steps = (sched_warmup_epochs + sched_cosine_epochs) * n_batches_per_epoch
    scheduler = _make_warmup_cosine_scheduler(
        opt,
        enabled=bool(scheduler_cfg.get("enabled", False)),
        total_steps=sched_total_steps,
        warmup_steps=sched_warmup_epochs * n_batches_per_epoch,
        min_lr_ratio=float(scheduler_cfg.get("min_lr_ratio", 0.0)),
    )
    if scheduler is not None:
        print(
            "    LR scheduler enabled for full_finetune: warmup_cosine "
            f"warmup_epochs={sched_warmup_epochs} "
            f"cosine_epochs={sched_cosine_epochs} "
            f"min_lr_ratio={float(scheduler_cfg.get('min_lr_ratio', 0.0)):g}"
        )

    cls_weights = None
    if class_balanced_loss:
        if train_labels is None:
            # Fallback: scan the train dataloader once to collect labels.
            ys: List[torch.Tensor] = []
            for batch in train_dl:
                yb = batch[1]
                ys.append(yb if torch.is_tensor(yb) else torch.as_tensor(yb))
            train_labels = torch.cat(ys, dim=0) if ys else torch.zeros(0, dtype=torch.long)
        cls_weights = _compute_class_weights(
            train_labels.long(), num_classes, scheme=class_weight_scheme
        ).to(device)
        print(
            f"    class_balanced_loss=True scheme={class_weight_scheme} "
            f"weights[min/mean/max]={cls_weights.min().item():.3f}/"
            f"{cls_weights.mean().item():.3f}/{cls_weights.max().item():.3f}"
        )
    loss_fn = nn.CrossEntropyLoss(weight=cls_weights)

    encode_head = _build_encode_head(module, head, device)

    best_val = float("inf")
    best_state = None
    best_epoch: int | None = None
    train_hist: List[float] = []
    val_hist: List[float] = []

    amp_enabled = bool(use_amp) and (device.type == "cuda")
    if amp_enabled:
        print(f"    AMP enabled for full_finetune (dtype={amp_dtype})")
    else:
        print("    AMP disabled for full_finetune (fp32)")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    # Prefer the newer torch.amp.GradScaler API when available.
    try:
        scaler = torch.amp.GradScaler(
            autocast_device_type,
            enabled=amp_enabled and amp_dtype == torch.float16,
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    es_cfg = early_stopping or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = es_cfg.get("patience", None)
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_warmup = int(es_cfg.get("warmup_epochs", 0))
    # Hard floor: never early-stop before this many epochs have completed,
    # regardless of patience/warmup.
    es_min_epochs = int(es_cfg.get("min_epochs", 0))

    if es_patience is None:
        es_enabled = False
    es_patience = int(es_patience) if es_enabled else 0
    bad_epochs = 0

    t0 = time.perf_counter()
    for epoch in range(int(epochs)):
        epoch_t0 = time.perf_counter()
        module.train()
        head.train()
        losses = []

        for batch in train_dl:
            x, y = batch[0].to(device, non_blocking=True).float(), batch[1].to(device, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=amp_enabled, device_type=autocast_device_type, dtype=amp_dtype):
                logits = encode_head(x)
                loss = loss_fn(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
                scaler.step(opt)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            else:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm))
                opt.step()
                if scheduler is not None:
                    scheduler.step()
            losses.append(loss.item())
            del x, y, logits, loss

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_hist.append(train_loss)

        # validation
        module.eval()
        head.eval()
        vlosses = []
        with torch.no_grad():
            for batch in val_dl:
                x, y = batch[0].to(device, non_blocking=True).float(), batch[1].to(device, non_blocking=True).long()
                with torch.amp.autocast(enabled=amp_enabled, device_type=autocast_device_type, dtype=amp_dtype):
                    logits = encode_head(x)
                    loss = loss_fn(logits, y)
                vlosses.append(loss.item())
                del x, y, logits, loss

        val_loss = float(np.mean(vlosses)) if vlosses else float("inf")
        val_hist.append(val_loss)

        # Periodic progress print (keeps SLURM logs alive and gives ETA)
        if (epoch + 1) == 1 or (epoch + 1) % 5 == 0 or (epoch + 1) == int(epochs):
            epoch_s = time.perf_counter() - epoch_t0
            total_s = time.perf_counter() - t0
            done = epoch + 1
            avg_s = total_s / max(1, done)
            eta_s = avg_s * max(0, int(epochs) - done)
            print(
                f"    Epoch {done}/{int(epochs)}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"epoch_time={epoch_s:.1f}s, eta~{eta_s/60.0:.1f}m",
                flush=True,
            )

        if val_loss < (best_val - es_min_delta):
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                "module": {k: v.detach().cpu() for k, v in module.state_dict().items()},
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            }
        else:
            bad_epochs += 1

        if (
            es_enabled
            and (epoch + 1) > es_warmup
            and (epoch + 1) >= es_min_epochs
            and bad_epochs >= es_patience
        ):
            print(
                f"    EarlyStopping: stop at epoch {epoch+1}/{int(epochs)} (monitor=val_loss, best={best_val:.6f}, "
                f"patience={es_patience}, min_delta={es_min_delta}, "
                f"warmup={es_warmup}, min_epochs={es_min_epochs})",
                flush=True,
            )
            break

        # periodic cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # restore best
    if best_state is not None:
        module.load_state_dict(best_state["module"], strict=False)
        head.load_state_dict(best_state["head"], strict=True)

    module.eval()
    head.eval()

    fit_info: Dict[str, Any] = {
        "monitor": "val_loss",
        "best_epoch": (int(best_epoch) + 1) if best_epoch is not None else None,
        "best_val_loss": float(best_val),
        "epochs_ran": int(len(train_hist)),
        "early_stopping": {
            "enabled": bool(es_enabled),
            "patience": int(es_patience) if es_enabled else None,
            "min_delta": float(es_min_delta),
            "warmup_epochs": int(es_warmup),
            "min_epochs": int(es_min_epochs),
        },
        "scheduler": {
            "enabled": scheduler is not None,
            "type": "warmup_cosine" if scheduler is not None else None,
            "warmup_epochs": sched_warmup_epochs,
            "cosine_epochs": sched_cosine_epochs if scheduler is not None else None,
            "min_lr_ratio": float(scheduler_cfg.get("min_lr_ratio", 0.0)),
        },
    }

    return head, train_hist, val_hist, fit_info


def _build_encode_head_for_eval(module: Any, head: nn.Module, device: torch.device) -> nn.Module:
    class _EncodeHead(nn.Module):
        def __init__(self, bb: Any, hd: nn.Module):
            super().__init__()
            self.backbone = bb
            self.head = hd

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.backbone.encode(x)
            return self.head(z)

    return _EncodeHead(module, head).to(device)


@torch.no_grad()
def _predict_probs(head: nn.Module, Z: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return softmax probabilities for each class."""
    head.eval()
    probs = []
    bs = 2048
    for i in range(0, Z.shape[0], bs):
        Zb = Z[i:i+bs].to(device)
        logits = head(Zb)
        pb = torch.softmax(logits.float(), dim=1).cpu()
        probs.append(pb)
        del Zb, logits, pb
    return torch.cat(probs, dim=0)

@torch.no_grad()
def _predict_probs_aggregate_mean(
    module: Any,
    head: nn.Module,
    dl: DataLoader,
    device: torch.device,
    base_window_size: int,   # in samples
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each batch of long windows x: (B, C, L_long),
    split into k subwindows of length base_window_size, run model on each,
    average probs over k => (B, num_classes).
    Returns: (probs, y_true)
    """
    module.eval()
    head.eval()

    all_probs = []
    all_y = []

    use_amp = False

    encode_head = _build_encode_head_for_eval(module, head, device=device)

    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    for batch in tqdm(dl, total=len(dl)):
        x_long = batch[0].to(device, non_blocking=True).float()  # (B,C,L)
        y = batch[1]
        B, C, L = x_long.shape

        k = L // base_window_size
        if k <= 0:
            continue

        L_use = k * base_window_size
        if L_use != L:
            x_long = x_long[..., :L_use]

        # (B,C,k,base) -> (B*k,C,base)
        x_sub = x_long.view(B, C, k, base_window_size).permute(0, 2, 1, 3).contiguous()
        x_sub = x_sub.view(B * k, C, base_window_size)

        with torch.amp.autocast(enabled=use_amp, device_type=autocast_device_type):
            logits = encode_head(x_sub)
        p_sub = torch.softmax(logits.float(), dim=1)  # (B*k, num_classes)

        p_sub = p_sub.view(B, k, -1).mean(dim=1)  # (B, num_classes)

        all_probs.append(p_sub.cpu())
        all_y.append(torch.as_tensor(y).long().cpu())

        del x_long, x_sub, logits, p_sub

    if not all_probs:
        raise ValueError(
            "aggregate_mean produced no batches. This usually means the provided windows are shorter than "
            f"base_window_size={base_window_size} (so k=L//base_window_size is 0 for all samples), or the "
            "test split is empty after filtering."
        )

    probs = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy()
    return probs, y_true

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, num_classes: int) -> Dict[str, float]:
    """Compute classification metrics. Uses sklearn when available; otherwise returns accuracy/f1 (micro) only."""
    out: Dict[str, float] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out["acc"] = float((y_true == y_pred).mean())

    out["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # AUC metrics require probabilities
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
            # Malformed probabilities; keep AUC metrics unset.
            return out

        # If some rows have non-finite probabilities (NaN/Inf), sklearn AUC will raise.
        # Keep the core classification metrics above computed on all rows, but compute
        # AUC/AUPRC on the subset with finite probability vectors.
        finite_mask = np.isfinite(y_prob).all(axis=1)
        if not bool(np.all(finite_mask)):
            y_true_auc = y_true[finite_mask]
            y_prob_auc = y_prob[finite_mask]
        else:
            y_true_auc = y_true
            y_prob_auc = y_prob

        # Some downstream splits can be missing classes. For AUROC, compute over only
        # the classes actually present in y_true (OVR macro across those present classes).
        present = np.unique(y_true_auc).astype(int)
        # Defensive: ignore negative labels and labels outside probability columns.
        present = present[(present >= 0) & (present < int(y_prob_auc.shape[1]))]

        if num_classes == 2:
            # Binary: only defined if both classes appear.
            if present.size < 2:
                out["auc_roc"] = float("nan")
                out["auc_pr"] = float("nan")
            else:
                pos = y_prob_auc[:, 1]
                out["auc_roc"] = float(roc_auc_score(y_true_auc, pos))
                out["auc_pr"] = float(average_precision_score(y_true_auc, pos))
        else:
            if present.size < 2:
                out["auc_roc_ovr_macro"] = float("nan")
                out["auc_pr_macro"] = float("nan")
            else:
                # Restrict probabilities to present classes only.
                y_prob_sub = y_prob_auc[:, present]

                # AUROC (OVR macro) across present classes.
                aucs: List[float] = []
                for j in range(len(present)):
                    y_true_bin = (y_true_auc == present[j]).astype(int)
                    try:
                        aucs.append(float(roc_auc_score(y_true_bin, y_prob_sub[:, j])))
                    except ValueError:
                        # Should be rare (e.g., all-0 or all-1 after filtering), but be safe.
                        continue
                out["auc_roc_ovr_macro"] = float(np.mean(aucs)) if len(aucs) else float("nan")

                # Keep PR AUC behavior aligned: macro over present classes.
                aps: List[float] = []
                for j in range(len(present)):
                    y_true_bin = (y_true_auc == present[j]).astype(int)
                    try:
                        aps.append(float(average_precision_score(y_true_bin, y_prob_sub[:, j])))
                    except ValueError:
                        continue
                out["auc_pr_macro"] = float(np.mean(aps)) if len(aps) else float("nan")

                # Parity with motion_supervised_baseline: many downstream consumers expect
                # auc_roc/auc_pr columns even for multiclass datasets.
                # Surface the OVR-macro values there.
                if np.isfinite(out.get("auc_roc_ovr_macro", float("nan"))):
                    out.setdefault("auc_roc", float(out["auc_roc_ovr_macro"]))
                if np.isfinite(out.get("auc_pr_macro", float("nan"))):
                    out.setdefault("auc_pr", float(out["auc_pr_macro"]))

    return out


def _plot_confusion(cm: np.ndarray, class_names: List[str] | None, title: str, out_path: Path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    # annotate
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_results_table(results: Dict[str, Any], save_dir: Path, csv_name: str = "linear_probe_results.csv"):
    """Save results as CSV and formatted text table."""
    rows = []
    for dataset_name, res in results["linear_probe"].items():
        fit = (res.get("fit", {}) or {}) if isinstance(res, dict) else {}
        row = {
            "dataset": dataset_name,
            "accuracy": res.get("acc", float("nan")),
            "balanced_accuracy": res.get("balanced_acc", float("nan")),
            "f1_macro": res.get("f1_macro", float("nan")),
            "f1_weighted": res.get("f1_weighted", float("nan")),
            "precision_macro": res.get("precision_macro", float("nan")),
            "recall_macro": res.get("recall_macro", float("nan")),
            "auc_roc": res.get("auc_roc", res.get("auc_roc_ovr_macro", float("nan"))),
            "auc_pr": res.get("auc_pr", res.get("auc_pr_macro", float("nan"))),
            "n_samples": res.get("n", 0),
            "n_classes": res.get("num_classes", 0),
            "train_n": res.get("train_n", 0),
            "val_n": res.get("val_n", 0),
            "test_n": res.get("test_n", 0),
            "best_epoch": fit.get("best_epoch", None),
            "epochs_ran": fit.get("epochs_ran", None),
            "error": res.get("error", ""),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by accuracy (descending), with errors at bottom
    df_valid = df[df["error"] == ""].copy()
    df_error = df[df["error"] != ""].copy()
    df_valid = df_valid.sort_values("accuracy", ascending=False)
    df = pd.concat([df_valid, df_error], ignore_index=True)
    
    # Save CSV
    csv_path = save_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results table to {csv_path}")
    
    # Save formatted text table
    txt_path = save_dir / "linear_probe_results.txt"
    # Some cluster environments default to ASCII locales; force UTF-8.
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("LINEAR PROBE EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Metadata
        f.write(f"Sampling Rate: {results['hz']} Hz\n")
        f.write(f"Window Duration: {results['window_sec']} seconds\n")
        f.write(f"Axial Mode: {results['axial_mode']}\n")
        f.write(f"Timestamp: {results.get('timestamp', 'unknown')}\n")
        f.write("\n" + "-"*80 + "\n\n")
        
        # Results table — include the metrics we already compute and persist to CSV
        # so the txt report alone is enough to skim a run without opening the CSV.
        header_cols = (
            f"{'Dataset':<25} {'Acc':>8} {'BalAcc':>8} {'F1m':>8} {'F1w':>8} "
            f"{'AUROC':>8} {'AUPR':>8} {'NumCls':>7} {'Test':>8} {'BestEp':>7} {'Status':>8}\n"
        )
        f.write(header_cols)
        f.write("-"*120 + "\n")

        def _fmt(v, w=8, p=4):
            try:
                fv = float(v)
                if fv != fv:  # NaN
                    return f"{'-':>{w}}"
                return f"{fv:>{w}.{p}f}"
            except Exception:
                return f"{'-':>{w}}"

        for _, row in df.iterrows():
            dataset = row["dataset"]
            n_cls = int(row.get("n_classes", 0) or 0)
            test_n = int(row.get("test_n", 0) or 0)
            best_ep = row.get("best_epoch", None)
            try:
                best_ep_f = float(best_ep)
                best_ep_str = "-" if not np.isfinite(best_ep_f) else str(int(best_ep_f))
            except Exception:
                best_ep_str = "-"
            if row["error"]:
                f.write(
                    f"{dataset:<25} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} "
                    f"{'N/A':>8} {'N/A':>8} {n_cls:>7} {test_n:>8} {best_ep_str:>7} {'ERROR':>8}\n"
                )
            else:
                f.write(
                    f"{dataset:<25} "
                    f"{_fmt(row.get('accuracy'))} "
                    f"{_fmt(row.get('balanced_accuracy'))} "
                    f"{_fmt(row.get('f1_macro'))} "
                    f"{_fmt(row.get('f1_weighted'))} "
                    f"{_fmt(row.get('auc_roc'))} "
                    f"{_fmt(row.get('auc_pr'))} "
                    f"{n_cls:>7} {test_n:>8} {best_ep_str:>7} {'OK':>8}\n"
                )

        f.write("\n" + "-"*120 + "\n\n")

        # Summary statistics: report mean/median across valid datasets for each metric.
        valid_df = df[df["error"] == ""]
        if len(valid_df) > 0:
            f.write("SUMMARY STATISTICS (across valid datasets)\n")
            f.write("-"*120 + "\n")
            f.write(f"{'metric':<22} {'mean':>10} {'median':>10} {'min':>10} {'max':>10} {'std':>10}\n")
            for col in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc_roc", "auc_pr"):
                if col not in valid_df.columns:
                    continue
                series = valid_df[col].astype(float)
                f.write(
                    f"{col:<22} {series.mean():>10.4f} {series.median():>10.4f} "
                    f"{series.min():>10.4f} {series.max():>10.4f} {series.std():>10.4f}\n"
                )
            f.write(f"\nSuccessful Datasets: {len(valid_df)} / {len(df)}\n")

        f.write("\n" + "="*120 + "\n")
    
    print(f"✓ Saved formatted table to {txt_path}")
    
    return df

def _should_strict_load_backbone(cfg: Dict[str, Any]) -> bool:
    # Both shipped methods (ar_transformer, patchtst) map cleanly onto their
    # pretraining checkpoints, so a strict state_dict load is appropriate.
    return True


def _load_backbone_from_path(backbone, ckpt_path, *, strict: bool = False, method: str = ""):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    backbone.load_state_dict(state, strict=strict)
    return backbone

def _save_metadata(metadata: Dict[str, Any], save_dir: Path):
    """Save experiment metadata as JSON."""
    json_path = save_dir / "linear_probe_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {json_path}")


def _plot_results(df: pd.DataFrame, save_dir: Path):
    """Create visualization plots."""
    
    df_valid = df[df["error"] == ""].copy()
    if len(df_valid) == 0:
        print("⚠ No valid results to plot")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Accuracy bar chart
    ax1 = plt.subplot(2, 2, 1)
    df_valid_sorted = df_valid.sort_values("accuracy", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_valid_sorted)))
    ax1.barh(range(len(df_valid_sorted)), df_valid_sorted["accuracy"], color=colors)
    ax1.set_yticks(range(len(df_valid_sorted)))
    ax1.set_yticklabels(df_valid_sorted["dataset"], fontsize=8)
    ax1.set_xlabel("Accuracy")
    ax1.set_title("Linear Probe Accuracy by Dataset")
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(df_valid["accuracy"].mean(), color='red', linestyle='--', label=f'Mean: {df_valid["accuracy"].mean():.3f}')
    ax1.legend()
    
    # 2. Accuracy distribution
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(df_valid["accuracy"], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(df_valid["accuracy"].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(df_valid["accuracy"].median(), color='orange', linestyle='--', linewidth=2, label='Median')
    ax2.set_xlabel("Accuracy")
    ax2.set_ylabel("Count")
    ax2.set_title("Accuracy Distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Samples vs Accuracy scatter
    ax3 = plt.subplot(2, 2, 3)
    scatter = ax3.scatter(df_valid["n_samples"], df_valid["accuracy"], 
                         c=df_valid["n_classes"], s=100, alpha=0.6, cmap='plasma')
    ax3.set_xlabel("Number of Samples")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Accuracy vs Dataset Size")
    ax3.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Number of Classes')
    
    # Add dataset labels for outliers
    for _, row in df_valid.iterrows():
        if row["accuracy"] > 0.9 or row["accuracy"] < 0.5:
            ax3.annotate(row["dataset"], (row["n_samples"], row["accuracy"]), 
                        fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    # 4. Classes vs Accuracy
    ax4 = plt.subplot(2, 2, 4)
    scatter2 = ax4.scatter(df_valid["n_classes"], df_valid["accuracy"], 
                          s=df_valid["n_samples"]/10, alpha=0.6, c=df_valid["accuracy"], cmap='RdYlGn')
    ax4.set_xlabel("Number of Classes")
    ax4.set_ylabel("Accuracy")
    ax4.set_title("Accuracy vs Number of Classes")
    ax4.grid(alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax4)
    cbar2.set_label('Accuracy')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = save_dir / "linear_probe_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plots to {plot_path}")


def _plot_training_curves(
    dataset_histories: Dict[str, Tuple[List, List]],
    save_dir: Path,
    *,
    finetune_mode: str = "linear_probe",
):
    """Plot training curves (train/val loss) for each dataset.

    `finetune_mode` is used only to label the output filename.
    """
    
    datasets_to_plot = list(dataset_histories.keys())
    if not datasets_to_plot:
        return
    
    n_datasets = len(datasets_to_plot)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for idx, dataset_name in enumerate(datasets_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        train_hist, val_hist = dataset_histories[dataset_name]
        epochs = range(1, len(train_hist) + 1)
        
        ax.plot(epochs, train_hist, label='Train Loss', marker='o', markersize=3)
        ax.plot(epochs, val_hist, label='Val Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(dataset_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    safe_mode = str(finetune_mode).replace("/", "_")
    curves_path = save_dir / f"finetune_loss_curves__{safe_mode}.png"
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {curves_path}")


def run_motion_linear_probe(cfg: Dict[str, Any], module: Any, ckpt_path: str, out_dir: str | None = None) -> Dict[str, Any]:
    """
    Run per-dataset linear-probe evaluation on MotionFM downstream datasets.
    
    Saves results to cfg["checkpointing"]["run_dir"] including:
    - CSV and formatted text tables
    - Metadata JSON
    - Visualization plots
    - Training curves
    """
    ecfg = (cfg.get("eval_motion", {}) or {})

    # Determine save directory
    if out_dir is None:
        out_dir = cfg.get("checkpointing", {}).get("run_dir", "outputs/eval")
    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"Saving evaluation results to: {save_dir}")
    print(f"{'='*80}\n")

    py_path = ecfg.get("dataloader_py", "motion_dataloader.py")
    motion = _load_motion_module(py_path)

    data_root = ecfg["data_root"]
    hz, window_sec = _infer_hz_and_window(cfg)
    window_size = int(round(hz * window_sec))
    axial_mode = _infer_axial_mode(cfg)

    extra_secs = ecfg.get("extra_eval_window_secs", []) or []
    # Extra eval is meant for *larger* windows where we can aggregate multiple base windows.
    # E.g., base=30s with extra=[60] is valid. base=60s with extra=[30] is not.
    extra_secs = [
        int(s)
        for s in extra_secs
        if int(s) > 0 and int(s) != int(window_sec) and int(s) > int(window_sec)
    ]
    extra_agg = str(ecfg.get("extra_eval_agg", "mean_prob"))
    save_prefix = str(ecfg.get("extra_eval_save_prefix", "linear_probe_results"))
    extra_results_by_sec = {s: {} for s in extra_secs}

    split_dirs_by_window_sec = ecfg.get("split_dirs_by_window_sec", None)  # dict like {"10": "...", "30": "..."}
    def _split_dir_for(sec: int):
        if isinstance(split_dirs_by_window_sec, dict):
            if str(sec) in split_dirs_by_window_sec:
                return split_dirs_by_window_sec[str(sec)]
        return ecfg.get("split_path_or_dir", None)

    lp = (ecfg.get("linear_probe", {}) or {})
    epochs = int(lp.get("epochs", 20))
    lr = float(lp.get("lr", 1e-3))
    wd = float(lp.get("weight_decay", 0.0))
    seed = int(lp.get("seed", 0))
    train_frac = float(lp.get("train_frac", 0.7))
    val_frac = float(lp.get("val_frac", 0.15))
    train_subsample_ratio = float(lp.get("train_subsample_ratio", 1.0))
    if not (0.0 < train_subsample_ratio <= 1.0):
        raise ValueError(f"eval_motion.linear_probe.train_subsample_ratio must be in (0,1], got {train_subsample_ratio}")

    # Finetune config:
    # - preferred: eval_motion.finetune.* (specific to MotionFM evaluation)
    # - fallback: run.finetune.* (older configs/scripts)
    ft = (ecfg.get("finetune", None) or (cfg.get("run", {}) or {}).get("finetune", {}) or {})
    finetune_mode = str(ft.get("mode", "linear_probe"))
    ft_epochs = int(ft.get("epochs") or epochs)
    ft_lr = float(ft.get("lr") or lr)
    ft_weight_decay = float(ft.get("weight_decay") or wd)
    ft_max_grad_norm = ft.get("max_grad_norm", 1.0)
    method_name = str((cfg.get("method", {}) or {}).get("name", "")).lower()
    ft_scheduler = dict((ft.get("scheduler", {}) or {}) if isinstance(ft, dict) else {})

    # Class-balanced cross-entropy. Default ON for both LP and FF so all
    # downstream datasets are trained with the same loss formulation. FF
    # falls back to LP setting unless the user overrides it explicitly.
    lp_class_balanced = bool(lp.get("class_balanced_loss", True))
    lp_class_weight_scheme = str(lp.get("class_weight_scheme", "inv_freq"))
    ft_class_balanced = bool(ft.get("class_balanced_loss", lp_class_balanced))
    ft_class_weight_scheme = str(ft.get("class_weight_scheme", lp_class_weight_scheme))

    bs = int(lp.get("batch_size", 256))
    # FF holds an autograd graph, so it usually needs a smaller batch than LP.
    # Falls back to LP batch size if not explicitly set.
    ft_bs = int(ft.get("batch_size", bs))
    preload = bool(ecfg.get("preload", False))

    # Workers + preload are *not* mutually exclusive on Linux: fork() gives each
    # worker a COW view of the preloaded `_sample_cache`, so we get parallel batch
    # assembly without duplicating memory. We also keep workers alive across epochs
    # to avoid re-importing the (heavy) motion_dataloader module per epoch.
    num_workers = int(lp.get("num_workers", 4))
    pin_memory = bool(lp.get("pin_memory", True))
    persistent_workers = bool(lp.get("persistent_workers", num_workers > 0))
    prefetch_factor_cfg = lp.get("prefetch_factor", None)
    prefetch_factor = int(prefetch_factor_cfg) if (prefetch_factor_cfg is not None and num_workers > 0) else None

    def _loader_kwargs(*, shuffle: bool, batch_size: int | None = None) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(
            batch_size=int(batch_size if batch_size is not None else bs),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        if prefetch_factor is not None and num_workers > 0:
            kw["prefetch_factor"] = prefetch_factor
        return kw

    # Early stopping knobs
    lp_es = (lp.get("early_stopping", {}) or {})
    ft_es = ((ft.get("early_stopping", {}) or {}) if isinstance(ft, dict) else {})
    # If user only specified eval_motion.linear_probe.early_stopping.*, reuse it for finetune.
    if not ft_es and lp_es:
        ft_es = dict(lp_es)

    flush_every_dataset = bool(ecfg.get("flush_every_dataset", True))
    flush_plots_every_dataset = bool(ecfg.get("flush_plots_every_dataset", False))

    # split_path_or_dir = (
    #     ecfg.get("split_path_or_dir", None)
    #     or ecfg.get("split_dir", None)
    #     or ecfg.get("frozen_split_dir", None)
    # )
    split_path_or_dir = _split_dir_for(int(window_sec))
    allow_oob_splits = bool(ecfg.get("allow_oob_splits", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mixed precision control for full_finetune. We intentionally key off
    # cfg.trainer.precision so sbatch overrides control the actual FF training
    # loop, not just the Lightning trainer config.
    tcfg = (cfg.get("trainer", {}) or {})
    prec = str(tcfg.get("precision", "32-true")).lower()
    amp_enabled = ("16" in prec) or ("bf16" in prec)
    if "bf16" in prec:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    module = module.to(device)
    module.eval()

    # Collect metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_info": _get_model_info(module, cfg),
        "data_config": {
            "hz": hz,
            "window_sec": window_sec,
            "window_size": window_size,
            "axial_mode": axial_mode,
            "sensor_types": ecfg.get("sensor_types", "accelerometer"),
            "placement": ecfg.get("placement", None),
            "top_k_labels": ecfg.get("top_k_labels", None),
        },
        "linear_probe_config": {
            "mode": finetune_mode,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": wd,
            "batch_size": bs,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "train_subsample_ratio": train_subsample_ratio,
            "seed": seed,
        },
        "finetune_config": {
            "mode": finetune_mode,
            "epochs": ft_epochs,
            "learning_rate": ft_lr,
            "weight_decay": ft_weight_decay,
            "max_grad_norm": ft_max_grad_norm,
            "scheduler": ft_scheduler,
        },
        "dataset_limits": {
            "max_files_per_dataset": ecfg.get("max_files_per_dataset", None),
            "max_windows": ecfg.get("max_windows", None),
            "subsample_ratio": ecfg.get("subsample_ratio", None),
            "capture24_split_max_windows_per_class": ecfg.get(
                "capture24_split_max_windows_per_class", None
            ),
        }
    }

    if split_path_or_dir is not None:
        metadata["split_config"] = {
            "mode": "frozen",
            "split_path_or_dir": str(split_path_or_dir),
            "allow_oob_splits": allow_oob_splits,
        }
    else:
        metadata["split_config"] = {
            "mode": "on_the_fly_stratified",
            "train_frac": train_frac,
            "val_frac": val_frac,
            "seed": seed,
        }

    results = {}
    dataset_histories = {}  # Store training curves (train_loss, val_loss) per dataset
    dataset_keys = list(getattr(motion, "DATASET_CONFIGS").keys())
    include_keys = ecfg.get("datasets", None) or ecfg.get("dataset_keys", None)
    exclude_keys = ecfg.get("exclude_datasets", None) or ecfg.get("exclude_dataset_keys", None)

    if include_keys is not None:
        include_set = {str(x) for x in include_keys}
        dataset_keys = [k for k in dataset_keys if str(k) in include_set]

    if exclude_keys is not None:
        exclude_set = {str(x) for x in exclude_keys}
        dataset_keys = [k for k in dataset_keys if str(k) not in exclude_set]

    if not dataset_keys:
        raise ValueError(
            "No datasets selected for evaluation. Set eval_motion.datasets (or remove include/exclude filters)."
        )

    metadata["datasets"] = {
        "selected": list(dataset_keys),
        "include": list(include_keys) if include_keys is not None else None,
        "exclude": list(exclude_keys) if exclude_keys is not None else None,
    }
    
    for ds_idx, key in enumerate(dataset_keys):
        print(f"\n{'='*60}")
        print(f"[{ds_idx+1}/{len(dataset_keys)}] Evaluating {key}")
        print(f"{'='*60}")

        try:
            max_windows_per_class = None
            if key == "capture24":
                # Frozen split indices are defined on the full filtered window list.
                # Per-class downsampling must happen *after* split resolution; see
                # eval_motion.capture24_split_max_windows_per_class.
                if split_path_or_dir is None:
                    max_windows_per_class = 1000

            module = _load_backbone_from_path(
                module,
                ckpt_path,
                strict=_should_strict_load_backbone(cfg),
                method=method_name,
            )
            module = module.to(device)

            # Label filtering/encoding can be dataset-specific. By default we defer to MotionDataset's
            # internal per-dataset defaults (TOP_K_LABELS). Optionally override Recofit only.
            top_k_labels = ecfg.get("top_k_labels", None)
            if str(key) == "Recofit" and ecfg.get("top_k_labels_recofit", None) is not None:
                top_k_labels = ecfg.get("top_k_labels_recofit")

            requested_placement = ecfg.get("placement", None)
            dataset_mode = str(ecfg.get("mode", "nonoverlap"))
            dataset_overlap_stride = int(ecfg.get("overlap_stride_samples", 1))
            if str(key) in {"capture24", "capture24_willetts"} and split_path_or_dir is not None:
                try:
                    p = Path(split_path_or_dir)
                    split_hint_path = p
                    if p.is_dir():
                        split_hint_path = p / f"{key}_splits.json"
                    if split_hint_path.is_file() and split_hint_path.suffix.lower() == ".json":
                        split_payload_hint = json.loads(split_hint_path.read_text(encoding="utf-8"))
                        split_dataset = str(split_payload_hint.get("dataset", key))
                        if split_dataset == str(key):
                            split_meta_hint = split_payload_hint.get("metadata", {}) or {}
                            stride_hint = split_meta_hint.get(
                                "stride_samples",
                                split_payload_hint.get("stride_samples", None),
                            )
                            mode_hint = split_meta_hint.get("mode", None)
                            if mode_hint is not None:
                                dataset_mode = str(mode_hint)
                            elif bool(getattr(motion.DATASET_CONFIGS.get(str(key)), "force_nonoverlap", False)):
                                dataset_mode = "nonoverlap"
                            if stride_hint is not None:
                                stride_hint = int(stride_hint)
                                dataset_overlap_stride = stride_hint
                            print(
                                f"  {key} frozen split overrides dataset indexing: "
                                f"mode={dataset_mode}, stride_samples={dataset_overlap_stride}"
                            )
                except Exception as e:
                    print(f"  Warning: failed to inspect {key} split file metadata: {e}")
            # If placement is not explicitly set, but we're using frozen splits, we can still
            # enforce a per-dataset default placement *via the split indices*.
            if split_path_or_dir is not None and requested_placement is None:
                try:
                    p = Path(split_path_or_dir)
                    split_dir = p if p.is_dir() else p.parent
                    defaults_path = split_dir / "default_arm_placements.json"
                    if defaults_path.exists():
                        payload = json.loads(defaults_path.read_text(encoding="utf-8"))
                        defaults = payload.get("defaults", {}) if isinstance(payload, dict) else {}
                        if isinstance(defaults, dict) and key in defaults:
                            requested_placement = defaults.get(key)
                            if requested_placement is not None:
                                print(f"  Default placement from frozen splits: {key} -> {requested_placement}")
                except Exception as e:
                    print(f"  Warning: failed to load default placements from frozen splits: {e}")

            ds_placement = requested_placement
            placement_for_split = None
            if split_path_or_dir is not None and requested_placement is not None:
                # Frozen split indices can be defined against either:
                #  (A) an *unfiltered* dataset window indexing (legacy behavior), in which case we must
                #      NOT filter MotionDataset by placement and instead filter indices at split-time, OR
                #  (B) a placement-filtered dataset (e.g. default-placement-only splits), in which case
                #      we SHOULD construct MotionDataset with the desired placement.
                try:
                    p = Path(split_path_or_dir)
                    split_dir = p if p.is_dir() else p.parent

                    has_defaults = (split_dir / "default_arm_placements.json").exists()
                    has_by_place = (split_dir / f"{key}_splits_by_placement.json").exists()
                    has_only = False
                    if isinstance(requested_placement, str) and requested_placement.strip():
                        ptag = requested_placement.strip().lower()
                        has_only = (split_dir / f"{key}_{ptag}_only_splits.json").exists()

                    split_is_unfiltered_indexed = bool(has_defaults or has_by_place or has_only)
                except Exception:
                    split_is_unfiltered_indexed = False

                if split_is_unfiltered_indexed:
                    # Explicitly disable placement filtering in MotionDataset.
                    # (placement=None now defaults to config.default_placement for some datasets.)
                    ds_placement = []
                    placement_for_split = requested_placement

            ds = motion.MotionDataset(
                data_root=data_root,
                dataset_name=key,
                sampling_rate=float(hz),
                window_size=int(window_size),
                sensor_types=ecfg.get("sensor_types", "accelerometer"),
                axial_mode=axial_mode,
                placement=ds_placement,
                label_column=ecfg.get("label_column", None),
                top_k_labels=top_k_labels,
                mode=dataset_mode,
                overlap_stride_samples=dataset_overlap_stride,
                preload=preload,
                return_majority_label=bool(ecfg.get("return_majority_label", True)),
                max_files_per_dataset=ecfg.get("max_files_per_dataset", None),
                max_windows=ecfg.get("max_windows", None),
                subsample_ratio=ecfg.get("subsample_ratio", None),
                max_windows_per_class=max_windows_per_class
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"  Skipping {key}: {msg}")
            results[key] = {"acc": float("nan"), "n": 0, "num_classes": 0, "error": msg}
            try:
                if "ds" in locals() and hasattr(ds, 'clear_cache'):
                    ds.clear_cache()
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass
            continue

        if len(ds) == 0:
            print(f"  Skipping {key}: no valid windows after filtering")
            results[key] = {"acc": float("nan"), "n": 0, "num_classes": 0, "error": "no_valid_windows"}
            continue

        print(f"  Dataset size: {len(ds)} windows")

        if hasattr(ds, 'get_all_labels'):
            ys = ds.get_all_labels()
        else:
            ys = []
            for i in range(len(ds)):
                _, y = ds[i]
                ys.append(int(y))
            ys = np.asarray(ys, dtype=int)

        unique_labels = np.unique(ys)
        num_classes = len(unique_labels)
        
        print(f"  Unique labels: {unique_labels}")
        print(f"  Num classes: {num_classes}")
        
        if num_classes <= 1:
            print(f"  Skipping {key}: insufficient classes ({num_classes})")
            results[key] = {"acc": float("nan"), "n": len(ds), "num_classes": num_classes, "error": "insufficient_classes"}
            if hasattr(ds, 'clear_cache'):
                ds.clear_cache()
            del ds
            gc.collect()
            continue

        if split_path_or_dir is not None:
            if not hasattr(motion, "get_split_indices_for_dataset"):
                raise AttributeError(
                    "Frozen splits requested (eval_motion.split_path_or_dir set) but motion_dataloader.py "
                    "does not define get_split_indices_for_dataset()."
                )
            splits, payload = motion.get_split_indices_for_dataset(
                ds,
                split_path_or_dir,
                allow_oob=allow_oob_splits,
                placement=placement_for_split,
            )
            train_idx = splits.get("train", [])
            val_idx = splits.get("val", [])
            test_idx = splits.get("test", [])
            print(
                f"  Using frozen split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
                f"(from {split_path_or_dir})"
            )
            try:
                split_meta = payload.get("metadata", {}) or {}
                if split_meta:
                    print(f"  Frozen split metadata keys: {sorted(split_meta.keys())}")

                # Defensive checks: many confusing metric failures are caused by applying
                # non-overlap split indices (n_windows ~ nonoverlap_windows) to an overlap-mode
                # dataset (len(ds) much larger), or by stride mismatches.
                n_windows = payload.get("n_windows", None)
                if isinstance(n_windows, int) and n_windows > 0:
                    # If split was generated for a much smaller indexing scheme, warn loudly.
                    if len(ds) >= 10 * int(n_windows):
                        mode_now = str(dataset_mode)
                        print(
                            f"  ⚠ Split/dataset indexing mismatch? split n_windows={n_windows} but len(ds)={len(ds)} "
                            f"(eval_motion.mode={mode_now}). This often means a non-overlap split file is being applied "
                            "to an overlap-mode dataset; AUROC can be missing if the resulting test subset is single-class."
                        )

                stride_in_split = split_meta.get("stride_samples", None)
                stride_now = dataset_overlap_stride
                if stride_in_split is not None and stride_now is not None:
                    try:
                        if int(stride_in_split) != int(stride_now):
                            print(
                                f"  ⚠ Split stride mismatch: split stride_samples={int(stride_in_split)} vs "
                                f"eval_motion.overlap_stride_samples={int(stride_now)}. Indices may not align."
                            )
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            train_idx, val_idx, test_idx = _stratified_split(
                ys, seed=seed, train_frac=train_frac, val_frac=val_frac
            )

        # capture24 + frozen split: cap windows per class *within* each split so indices
        # stay aligned with the frozen file while limiting compute.
        cap_cfg = ecfg.get("capture24_split_max_windows_per_class", None)
        if (
            str(key) in {"capture24"}
            and split_path_or_dir is not None
            and cap_cfg is not None
            and int(cap_cfg) > 0
        ):
            cap_k = int(cap_cfg)
            global_classes = set(np.unique(ys).tolist())
            before = (len(train_idx), len(val_idx), len(test_idx))
            train_idx = _cap_indices_per_class_within_split(train_idx, ys, cap_k, seed + 11)
            val_idx = _cap_indices_per_class_within_split(val_idx, ys, cap_k, seed + 22)
            test_idx = _cap_indices_per_class_within_split(test_idx, ys, cap_k, seed + 33)
            after = (len(train_idx), len(val_idx), len(test_idx))
            print(
                f"  {key} per-split max_windows_per_class={cap_k}: "
                f"train {before[0]}->{after[0]}, val {before[1]}->{after[1]}, test {before[2]}->{after[2]}"
            )
            for tag, part in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
                miss = sorted(global_classes - _labels_present(part, ys))
                if miss:
                    print(
                        f"  ⚠ {key} {tag} split has no windows for label(s) {miss} "
                        "(inherited from frozen split; not caused by per-split capping)"
                    )
        
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"  Skipping {key}: insufficient data after split")
            results[key] = {"acc": float("nan"), "n": len(ds), "num_classes": num_classes, "error": "insufficient_split"}
            if hasattr(ds, 'clear_cache'):
                ds.clear_cache()
            del ds
            gc.collect()
            continue

        print(f"  Split sizes - train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
        # Optional: subsample training indices *after* split selection.
        # This is safe with frozen splits because it does not change dataset indexing.
        if train_subsample_ratio < 1.0:
            train_idx = _subsample_indices_stratified(
                train_idx,
                labels=ys,
                ratio=train_subsample_ratio,
                seed=seed,
            )
            print(
                f"  Train subsample ratio={train_subsample_ratio:g} -> train={len(train_idx)} (val={len(val_idx)} test={len(test_idx)})"
            )

        if len(val_idx) == 0 and len(train_idx) >= 2:
            # Some frozen split dirs can yield val=0 after placement filtering.
            # Carve a small, deterministic stratified validation set from train.
            target_n_val = int(round(0.1 * len(train_idx)))
            target_n_val = max(1, min(target_n_val, min(256, len(train_idx) - 1)))
            ratio = float(target_n_val) / float(len(train_idx))
            carved_val = _subsample_indices_stratified(
                train_idx,
                labels=ys,
                ratio=ratio,
                seed=seed + 999,
            )
            carved_set = set(carved_val)
            if carved_set:
                train_idx = [i for i in train_idx if i not in carved_set]
                val_idx = list(carved_val)
                print(
                    f"  Frozen val split empty; carved val from train -> train={len(train_idx)} val={len(val_idx)} (test={len(test_idx)})"
                )

        # CrossEntropyLoss requires targets in [0, num_classes-1].
        # MotionDataset labels are often non-zero-based (e.g. {1,2} for daphnet_fog),
        # so wrap subsets with a label-mapper.
        class _LabelMappedDataset(torch.utils.data.Dataset):
            def __init__(self, base, label_map):
                self.base = base
                self.label_map = label_map

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                x, y = self.base[idx]
                y_int = int(y)
                return x, int(self.label_map.get(y_int, 0))

        # Build a stable label map using labels present in this dataset.
        # We keep it deterministic (sorted) so runs are comparable.
        unique_all = np.unique(ys)
        label_map = {int(old): int(new) for new, old in enumerate(sorted(unique_all.tolist()))}
        num_classes = len(label_map)
        print(f"  Label mapping ({num_classes} classes): {label_map}")

        print(f"  Training ({finetune_mode})...")
        if finetune_mode == "linear_probe":
            # Encode embeddings and fit a linear head.
            dtrain = DataLoader(Subset(ds, train_idx), **_loader_kwargs(shuffle=False))
            dval = DataLoader(Subset(ds, val_idx), **_loader_kwargs(shuffle=False))
            dtest = DataLoader(Subset(ds, test_idx), **_loader_kwargs(shuffle=False))

            print(f"  Encoding training set...")
            Ztr, ytr = _encode_dataset(module, dtrain, device)
            print(f"    Train embeddings: {Ztr.shape}")

            print(f"  Encoding validation set...")
            Zv, yv = _encode_dataset(module, dval, device)
            print(f"    Val embeddings: {Zv.shape}")

            print(f"  Encoding test set...")
            Zte, yte = _encode_dataset(module, dtest, device)
            print(f"    Test embeddings: {Zte.shape}")

            ytr_mapped = torch.tensor([label_map.get(int(y), 0) for y in ytr])
            yv_mapped = torch.tensor([label_map.get(int(y), 0) for y in yv])
            yte_mapped = torch.tensor([label_map.get(int(y), 0) for y in yte])

            head, train_hist, val_hist, fit_info = _fit_linear_head(
                Ztr, ytr_mapped, Zv, yv_mapped,
                num_classes=num_classes, epochs=epochs, lr=lr,
                weight_decay=wd, seed=seed, device=device,
                batch_size=bs,
                early_stopping=lp_es,
                class_balanced_loss=lp_class_balanced,
                class_weight_scheme=lp_class_weight_scheme,
            )

            # We no longer need raw ds in linear probe mode.
            del dtrain, dval, dtest
            if hasattr(ds, "clear_cache"):
                ds.clear_cache()
            del ds
            gc.collect()
            torch.cuda.empty_cache()
        elif finetune_mode == "full_finetune":
            # Supervised finetune backbone + head on raw windows.
            dtrain_ft = DataLoader(
                _LabelMappedDataset(Subset(ds, train_idx), label_map),
                **_loader_kwargs(shuffle=True, batch_size=ft_bs),
            )
            dval_ft = DataLoader(
                _LabelMappedDataset(Subset(ds, val_idx), label_map),
                **_loader_kwargs(shuffle=False, batch_size=ft_bs),
            )
            # Pre-compute train labels for class weights so _fit_full_finetune
            # doesn't have to scan the dataloader (which would touch every window).
            train_labels_for_weights = None
            if ft_class_balanced:
                try:
                    ytr_ff = ys[np.asarray(train_idx, dtype=np.int64)]
                    train_labels_for_weights = torch.as_tensor(
                        np.asarray([label_map[int(v)] for v in ytr_ff], dtype=np.int64)
                    )
                except Exception:
                    train_labels_for_weights = None  # fallback: scan inside _fit_full_finetune

            head, train_hist, val_hist, fit_info = _fit_full_finetune(
                module, dtrain_ft, dval_ft,
                num_classes=num_classes, epochs=ft_epochs, lr=ft_lr,
                weight_decay=ft_weight_decay, seed=seed, device=device,
                use_amp=amp_enabled,
                amp_dtype=amp_dtype,
                max_grad_norm=ft_max_grad_norm,
                early_stopping=ft_es,
                class_balanced_loss=ft_class_balanced,
                class_weight_scheme=ft_class_weight_scheme,
                train_labels=train_labels_for_weights,
                scheduler_cfg=ft_scheduler,
            )
            del dtrain_ft, dval_ft
            for p in module.parameters():
                p.grad = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise ValueError(f"Unknown eval_motion.finetune.mode: {finetune_mode}")

        # Store training curves
        dataset_histories[key] = (train_hist, val_hist)


        
        print(f"  Evaluating...")
        # Predictions + probabilities
        if finetune_mode == "linear_probe":
            if not torch.isfinite(Zte).all():
                print("  ⚠ non-finite embeddings in Zte")
            probs = _predict_probs(head, Zte, device=device).numpy()
            if not np.isfinite(probs).all():
                bad = np.where(~np.isfinite(probs))
                print(f"  ⚠ non-finite probs: {len(bad[0])} entries; "
                      f"rows with any bad: {len(np.unique(bad[0]))}/{probs.shape[0]}")
            y_true = yte_mapped.numpy()
        else:
            # full finetune: run head on raw windows
            y_true_list = []
            probs_list = []
            dtest_ft = DataLoader(
                _LabelMappedDataset(Subset(ds, test_idx), label_map),
                **_loader_kwargs(shuffle=False, batch_size=ft_bs),
            )
            module.eval()
            head.eval()
            encode_head = _build_encode_head_for_eval(module, head, device=device)
            with torch.no_grad():
                for batch in dtest_ft:
                    x = batch[0].to(device, non_blocking=True).float()
                    yb = batch[1]
                    with torch.amp.autocast(enabled=(amp_enabled and device.type == "cuda"), device_type=("cuda" if device.type == "cuda" else "cpu"), dtype=amp_dtype):
                        logits = encode_head(x)
                    pb = torch.softmax(logits.float(), dim=1).cpu()
                    probs_list.append(pb)
                    y_true_list.append(torch.as_tensor(yb).long().cpu())
                    del x, yb, logits, pb
            probs = torch.cat(probs_list, dim=0).numpy()
            y_true = torch.cat(y_true_list, dim=0).numpy()
            del dtest_ft, y_true_list, probs_list, encode_head
            for p in module.parameters():
                p.grad = None
            if hasattr(ds, 'clear_cache'):
                ds.clear_cache()
            del ds
            gc.collect()
            torch.cuda.empty_cache()

        y_pred = probs.argmax(axis=1)
        metrics = _compute_metrics(y_true=y_true, y_pred=y_pred, y_prob=probs, num_classes=num_classes)
        acc = metrics.get("acc", float("nan"))
        print(f"  ✓ Test accuracy: {acc:.4f}")

        # Confusion matrix plot
        if num_classes <= 50:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
            ds_dir = save_dir / "per_dataset" / key.replace("/", "__")
            ds_dir.mkdir(parents=True, exist_ok=True)
            _plot_confusion(cm, None, f"{key} confusion matrix", ds_dir / "confusion_matrix.png")

        results[key] = {
            **metrics,
            "n": len(ys),
            "num_classes": num_classes,
            "train_n": len(train_idx),
            "val_n": len(val_idx),
            "test_n": len(test_idx),
            "fit": fit_info,
            }
        
        # ---- Extra eval on larger windows (aggregate mean prob) ----
        for big_sec in extra_secs:
            big_window_size = int(round(hz * big_sec))

            # Build a dataset that yields *big* windows
            ds_big = motion.MotionDataset(
                data_root=data_root,
                dataset_name=key,
                sampling_rate=float(hz),
                window_size=int(big_window_size),
                sensor_types=ecfg.get("sensor_types", "accelerometer"),
                axial_mode=axial_mode,
                placement=ecfg.get("placement", None),
                label_column=ecfg.get("label_column", None),
                top_k_labels=top_k_labels,
                mode=dataset_mode,
                overlap_stride_samples=dataset_overlap_stride,
                preload=preload,
                return_majority_label=bool(ecfg.get("return_majority_label", True)),
                max_files_per_dataset=ecfg.get("max_files_per_dataset", None),
                max_windows=ecfg.get("max_windows", None),
                subsample_ratio=ecfg.get("subsample_ratio", None),
                max_windows_per_class=max_windows_per_class,
            )

            if len(ds_big) == 0:
                extra_results_by_sec[big_sec][key] = {"acc": float("nan"), "n": 0, "num_classes": 0, "error": "no_valid_windows"}
                continue

            split_dir_big = _split_dir_for(int(big_sec))
            if split_dir_big is None:
                raise ValueError(f"No split dir provided for {big_sec}s. Set eval_motion.split_dirs_by_window_sec['{big_sec}'].")

            splits, payload = motion.get_split_indices_for_dataset(ds_big, split_dir_big, allow_oob=allow_oob_splits)
            train_idx_b = splits.get("train", [])
            val_idx_b = splits.get("val", [])
            test_idx_b = splits.get("test", [])

            n_big = len(ds_big)
            for name, idx in [("train", train_idx_b), ("val", val_idx_b), ("test", test_idx_b)]:
                if len(idx) == 0:
                    raise ValueError(f"{key} {big_sec}s: empty {name} split from {split_dir_big}")
                if np.max(idx) >= n_big or np.min(idx) < 0:
                    raise ValueError(f"{key} {big_sec}s: {name} split indices out of bounds for dataset len={n_big} (split dir {split_dir_big})")
    
            # Build label array for *this* dataset instance (needed for mapping + num_classes)
            if hasattr(ds_big, "get_all_labels"):
                ys_big = ds_big.get_all_labels()
            else:
                ys_big = np.asarray([int(ds_big[i][1]) for i in range(len(ds_big))], dtype=int)

            # IMPORTANT: Reuse the *base* label mapping and num_classes from the head we trained.
            # The head's output columns correspond to `label_map` (from the base-window dataset).
            # Creating a new mapping here can produce class indices >= head output dims and crash
            # AUROC computation (and is semantically incorrect).
            #
            # If the big-window split includes labels unseen in the base mapping, drop those
            # windows for this extra-eval (the model has no corresponding logits).
            base_label_keys = set(int(k) for k in label_map.keys())
            test_idx_b = [i for i in test_idx_b if int(ys_big[int(i)]) in base_label_keys]
            if len(test_idx_b) == 0:
                extra_results_by_sec[big_sec][key] = {
                    "acc": float("nan"),
                    "n": int(len(ys_big)),
                    "num_classes": int(num_classes),
                    "train_n": int(len(train_idx_b)),
                    "val_n": int(len(val_idx_b)),
                    "test_n": 0,
                    "error": "no_test_windows_after_label_filter",
                }
                if hasattr(ds_big, "clear_cache"):
                    ds_big.clear_cache()
                del ds_big
                gc.collect()
                torch.cuda.empty_cache()
                continue

            dtest_big = DataLoader(
                _LabelMappedDataset(Subset(ds_big, test_idx_b), label_map),
                **_loader_kwargs(shuffle=False),
            )

            # sanity: big window should be multiple of base window (else we crop)
            if extra_agg != "mean_prob":
                raise ValueError(f"Unsupported extra_eval_agg={extra_agg} (only mean_prob supported)")

            try:
                probs_b, y_true_b = _predict_probs_aggregate_mean(
                    module=module, head=head, dl=dtest_big, device=device, base_window_size=window_size
                )
                y_pred_b = probs_b.argmax(axis=1)
                metrics_b = _compute_metrics(
                    y_true=y_true_b, y_pred=y_pred_b, y_prob=probs_b, num_classes=num_classes
                )
                extra_results_by_sec[big_sec][key] = {
                    **metrics_b,
                    "n": int(len(ys_big)),
                    "num_classes": int(num_classes),
                    "train_n": int(len(train_idx_b)),
                    "val_n": int(len(val_idx_b)),
                    "test_n": int(len(test_idx_b)),
                }
            except Exception as e:
                extra_results_by_sec[big_sec][key] = {
                    "acc": float("nan"),
                    "n": int(len(ys_big)),
                    "num_classes": int(num_classes),
                    "train_n": int(len(train_idx_b)),
                    "val_n": int(len(val_idx_b)),
                    "test_n": int(len(test_idx_b)),
                    "error": f"{type(e).__name__}: {e}",
                }

            # cleanup big dataset
            del dtest_big
            if hasattr(ds_big, "clear_cache"): ds_big.clear_cache()
            del ds_big
            gc.collect()
            torch.cuda.empty_cache()

        if flush_every_dataset:
            try:
                partial = {
                    "linear_probe": results,
                    "hz": hz,
                    "window_sec": window_sec,
                    "axial_mode": axial_mode,
                    "timestamp": metadata["timestamp"],
                }
                _save_metadata(metadata, save_dir)
                df_partial = _save_results_table(partial, save_dir)
                if flush_plots_every_dataset:
                    _plot_results(df_partial, save_dir)
                    _plot_training_curves(dataset_histories, save_dir, finetune_mode=finetune_mode)
                print(f"  ✓ Flushed partial results after {key}")
            except Exception as e:
                print(f"  ⚠ Failed to flush partial results after {key}: {e}")

        # Cleanup (some tensors may not exist for full_finetune)
        if finetune_mode == "linear_probe":
            del Ztr, ytr, Zv, yv, Zte, yte
            del ytr_mapped, yv_mapped, yte_mapped
        del head
        gc.collect()
        torch.cuda.empty_cache()

    # Compile final results
    final_results = {
        "linear_probe": results,
        "hz": hz,
        "window_sec": window_sec,
        "axial_mode": axial_mode,
        "timestamp": metadata["timestamp"],
    }

    # Save all outputs
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Save metadata
    _save_metadata(metadata, save_dir)
    
    # Save results table
    df = _save_results_table(final_results, save_dir)

    for big_sec, res_map in extra_results_by_sec.items():
        extra_payload = {
            "linear_probe": res_map,
            "hz": hz,
            "window_sec": int(big_sec),
            "axial_mode": axial_mode,
            "timestamp": metadata["timestamp"],
        }
        _save_results_table(extra_payload, save_dir, csv_name=f"{save_prefix}__{big_sec}s.csv")
    
    # Create plots
    _plot_results(df, save_dir)
    _plot_training_curves(dataset_histories, save_dir, finetune_mode=finetune_mode)

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    for key, res in results.items():
        if "error" in res:
            print(f"{key:25s}: ERROR - {res['error']}")
        else:
            print(f"{key:25s}: {res['acc']:.4f} (n={res['n']}, classes={res['num_classes']})")
    
    # Overall statistics
    valid_accs = [res["acc"] for res in results.values() if "error" not in res]
    if valid_accs:
        print(f"\n{'='*80}")
        print(f"Mean Accuracy: {np.mean(valid_accs):.4f} ± {np.std(valid_accs):.4f}")
        print(f"Median Accuracy: {np.median(valid_accs):.4f}")
        print(f"Successful: {len(valid_accs)} / {len(results)} datasets")
        print(f"{'='*80}\n")

    return final_results