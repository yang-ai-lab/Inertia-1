"""Patient-level disease/medication detection via multiple-instance learning (MIL).

This is a *minimal, self-contained* version of the disease-detection evaluation
pipeline. It demonstrates how we probe a frozen pretrained backbone for
patient-level outcomes (e.g. depression severity, sleep complaints, medication
use) using attention-based MIL over a "bag" of accelerometer windows per
patient.

In our internal work this pipeline runs on NHANES accelerometry, which we
cannot redistribute. We therefore ship a generic, placeholder-data loader and
keep the patient/bag construction deliberately high level. See the README
("Disease detection") for the NHANES download pointer and for how the real
pipeline differs (per-day bags keyed on wear-time, fixed 24h grid, etc.).

Placeholder data layout
------------------------
Each patient is one ``.npy`` file containing that patient's accelerometer
windows as a float array of shape ``[N_windows, C, T]`` (C=axes, T=window
samples)::

    <data_root>/<patient_id>.npy

Labels CSV (``--labels-csv``): columns ``patient_id,label``.
Split  CSV (``--split-csv``):  columns ``patient_id,split`` with split in
{train, val, test}.

Supported backbones: ``ar_transformer``, ``patchtst`` (same as pretraining).

Example
-------
    python -m inertia1.scripts.eval_nhanes_labels \
        --data-root ./data/disease/windows \
        --labels-csv ./data/disease/labels.csv \
        --split-csv  ./data/disease/split.csv \
        --task depression --task-kind multiclass --num-classes 5 \
        --method ar_transformer --preset small \
        --ckpt-path /path/to/pretrained/last.ckpt \
        --bag-size 256 --epochs 20 --out-dir ./logs/disease
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from inertia1.methods.registry import build_experiment
from inertia1.run import apply_preset, infer_preset_name_from_paths, load_configs


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _maybe_import_sklearn_metrics() -> Optional[Dict[str, Any]]:
    try:
        from sklearn import metrics as skm
    except Exception:
        return None
    return {
        "accuracy_score": skm.accuracy_score,
        "balanced_accuracy_score": skm.balanced_accuracy_score,
        "f1_score": skm.f1_score,
        "precision_score": skm.precision_score,
        "recall_score": skm.recall_score,
        "roc_auc_score": skm.roc_auc_score,
        "average_precision_score": skm.average_precision_score,
    }


def _confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> List[List[int]]:
    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for t, p in zip(y_true.astype(int).tolist(), y_pred.astype(int).tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm.tolist()


# ---------------------------------------------------------------------------
# Placeholder patient-bag dataset
# ---------------------------------------------------------------------------
class PatientWindowBagDataset(Dataset):
    """Return one fixed-grid *bag* of windows per patient.

    Each patient's windows (``[N, C, T]``) are placed onto a fixed grid of
    ``bag_size`` slots spread evenly across the recording. Empty slots are
    zero-padded and flagged via ``valid_mask`` so the MIL pooler can learn a
    dedicated "missing" embedding. Every item returns:

        x_bag      [K, C, T]
        y          scalar (long for classification, float for regression)
        pos_idx    [K]   slot indices (for positional encoding)
        valid_mask [K]   1.0 if the slot holds a real window, else 0.0
    """

    def __init__(
        self,
        data_root: str | Path,
        patient_ids: List[str],
        label_map: Dict[str, float | int],
        *,
        kind: Literal["binary", "multiclass", "regression"],
        bag_size: int,
        seed: int = 0,
    ):
        self.data_root = Path(data_root)
        self.kind = str(kind)
        self.bag_size = int(bag_size)
        self.seed = int(seed)
        self.patient_ids = [str(p) for p in patient_ids if str(p) in label_map]
        self.label_map = label_map
        if not self.patient_ids:
            raise ValueError("No patients with both data and labels were found.")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _patient_path(self, pid: str) -> Path:
        return self.data_root / f"{pid}.npy"

    def __getitem__(self, idx: int):
        pid = self.patient_ids[int(idx)]
        windows = np.load(self._patient_path(pid)).astype(np.float32)  # [N, C, T]
        if windows.ndim != 3:
            raise ValueError(f"Patient {pid}: expected [N, C, T], got shape {windows.shape}")
        n = int(windows.shape[0])

        pos_idx = torch.arange(self.bag_size, dtype=torch.long)
        valid = torch.zeros(self.bag_size, dtype=torch.float32)
        bag = torch.zeros((self.bag_size, windows.shape[1], windows.shape[2]), dtype=torch.float32)

        if n > 0:
            # Evenly map the K grid slots onto the available windows (fixed grid).
            slot_to_win = np.floor(np.linspace(0, n, self.bag_size, endpoint=False)).astype(int)
            slot_to_win = np.clip(slot_to_win, 0, n - 1)
            for slot, w in enumerate(slot_to_win.tolist()):
                bag[slot] = torch.from_numpy(windows[w])
                valid[slot] = 1.0

        y = self.label_map[pid]
        y_t = torch.tensor(float(y), dtype=torch.float32) if self.kind == "regression" else torch.tensor(int(y), dtype=torch.long)
        return bag, y_t, pos_idx, valid


# ---------------------------------------------------------------------------
# MIL pooling (positional attention with a learned "missing slot" embedding)
# ---------------------------------------------------------------------------
def _sinusoidal_position_table(max_positions: int, dim: int) -> torch.Tensor:
    pos = torch.arange(int(max_positions), dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / float(dim)))
    pe = torch.zeros(int(max_positions), int(dim), dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class _PositionEmbedding(nn.Module):
    def __init__(self, dim: int, *, max_positions: int, encoding: Literal["learned", "sinusoidal"] = "learned"):
        super().__init__()
        self.encoding = str(encoding)
        if self.encoding == "learned":
            self.embedding = nn.Embedding(int(max_positions), int(dim))
            self.register_buffer("table", torch.empty(0), persistent=False)
        elif self.encoding == "sinusoidal":
            self.embedding = None
            self.register_buffer("table", _sinusoidal_position_table(max_positions, dim), persistent=True)
        else:
            raise ValueError(f"Unknown position encoding: {encoding}")

    @property
    def num_embeddings(self) -> int:
        return int(self.embedding.num_embeddings) if self.embedding is not None else int(self.table.shape[0])

    def forward(self, pos_idx: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            return self.embedding(pos_idx)
        return self.table.to(device=pos_idx.device)[pos_idx]


class PositionalMILAttentionPool(nn.Module):
    """Attention pooling over a fixed grid of instances with missing slots.

    Input:  Z [B, K, D] (+ optional pos_idx [B, K], valid_mask [B, K])
    Output: pooled [B, D], attn_weights [B, K]
    """

    def __init__(
        self,
        dim: int,
        *,
        max_positions: int = 4096,
        hidden: int = 128,
        dropout: float = 0.0,
        pos_encoding: Literal["learned", "sinusoidal"] = "learned",
    ):
        super().__init__()
        self.pos_emb = _PositionEmbedding(dim, max_positions=int(max_positions), encoding=pos_encoding)
        self.missing_emb = nn.Parameter(torch.zeros(int(dim)))
        self.input_dropout = nn.Dropout(p=float(dropout))
        self.attn = nn.Sequential(
            nn.Linear(int(dim), int(hidden)),
            nn.Tanh(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(
        self,
        Z: torch.Tensor,
        *,
        pos_idx: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, d = Z.shape
        if pos_idx is None:
            pos_idx = torch.arange(k, device=Z.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        pos_idx = pos_idx.to(device=Z.device, dtype=torch.long).clamp(0, self.pos_emb.num_embeddings - 1)
        pos = self.pos_emb(pos_idx).to(dtype=Z.dtype)
        Zp = Z + pos
        if valid_mask is not None:
            mask = valid_mask.to(device=Z.device, dtype=Zp.dtype).unsqueeze(-1)
            missing = self.missing_emb.to(dtype=Zp.dtype).view(1, 1, d) + pos
            Zp = Zp * mask + missing * (1.0 - mask)
        Zp = self.input_dropout(Zp)
        a = self.attn(Zp).squeeze(-1)  # [B, K]
        w = torch.softmax(a, dim=1)
        pooled = (Zp * w.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return pooled, w


class PatientMILClassifier(nn.Module):
    """Frozen backbone -> per-window embeddings -> MIL pool -> linear head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        embed_dim: int,
        num_outputs: int,
        bag_size: int,
        mil_hidden: int = 128,
        mil_dropout: float = 0.0,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():  # frozen feature extractor
            p.requires_grad = False
        self.pool = PositionalMILAttentionPool(
            embed_dim,
            max_positions=max(bag_size, 16),
            hidden=mil_hidden,
            dropout=mil_dropout,
            pos_encoding=pos_encoding,
        )
        self.head = nn.Linear(int(embed_dim), int(num_outputs))

    def encode_bag(self, x_bag: torch.Tensor) -> torch.Tensor:
        b, k, c, t = x_bag.shape
        flat = x_bag.reshape(b * k, c, t)
        with torch.no_grad():
            self.backbone.eval()
            z = self.backbone.encode(flat)  # [B*K, D]
        return z.reshape(b, k, -1)

    def forward(self, x_bag: torch.Tensor, *, pos_idx=None, valid_mask=None) -> torch.Tensor:
        z = self.encode_bag(x_bag)
        pooled, _ = self.pool(z, pos_idx=pos_idx, valid_mask=valid_mask)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    sk = _maybe_import_sklearn_metrics()
    y_pred = (y_prob >= 0.5).astype(int)
    out: Dict[str, Any] = {"confusion_matrix": _confusion_matrix_counts(y_true, y_pred, num_classes=2)}
    if sk is not None:
        out["accuracy"] = float(sk["accuracy_score"](y_true, y_pred))
        out["balanced_accuracy"] = float(sk["balanced_accuracy_score"](y_true, y_pred))
        out["f1"] = float(sk["f1_score"](y_true, y_pred, zero_division=0))
        out["precision"] = float(sk["precision_score"](y_true, y_pred, zero_division=0))
        out["recall"] = float(sk["recall_score"](y_true, y_pred, zero_division=0))
        if len(np.unique(y_true)) == 2:
            out["auroc"] = float(sk["roc_auc_score"](y_true, y_prob))
            out["auprc"] = float(sk["average_precision_score"](y_true, y_prob))
        else:
            out["auroc"] = float("nan")
            out["auprc"] = float("nan")
    else:
        out["accuracy"] = float(np.mean(y_true == y_pred))
    return out


def _metrics_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    sk = _maybe_import_sklearn_metrics()
    y_pred = np.argmax(y_prob, axis=1).astype(int)
    num_classes = int(y_prob.shape[1])
    out: Dict[str, Any] = {"confusion_matrix": _confusion_matrix_counts(y_true, y_pred, num_classes=num_classes)}
    if sk is not None:
        out["accuracy"] = float(sk["accuracy_score"](y_true, y_pred))
        out["balanced_accuracy"] = float(sk["balanced_accuracy_score"](y_true, y_pred))
        out["f1_macro"] = float(sk["f1_score"](y_true, y_pred, average="macro", zero_division=0))
        aucs, aps = [], []
        for c in np.unique(y_true).astype(int).tolist():
            y_c = (y_true == c).astype(int)
            if len(np.unique(y_c)) < 2 or c >= num_classes:
                continue
            try:
                aucs.append(float(sk["roc_auc_score"](y_c, y_prob[:, c])))
                aps.append(float(sk["average_precision_score"](y_c, y_prob[:, c])))
            except Exception:
                pass
        out["auroc_ovr_macro"] = float(np.mean(aucs)) if aucs else float("nan")
        out["auprc_ovr_macro"] = float(np.mean(aps)) if aps else float("nan")
    else:
        out["accuracy"] = float(np.mean(y_true == y_pred))
    return out


def _select_metric_value(metrics: Dict[str, Any], kind: str) -> float:
    key = "auprc" if kind == "binary" else "auroc_ovr_macro"
    val = metrics.get(key, float("nan"))
    return float(val) if val == val else 0.0  # NaN -> 0


# ---------------------------------------------------------------------------
# CSV / config helpers
# ---------------------------------------------------------------------------
def _read_two_col_csv(path: str | Path, value_cast) -> Dict[str, Any]:
    import csv

    out: Dict[str, Any] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if len(cols) < 2:
            raise ValueError(f"{path}: expected at least 2 columns, got {cols}")
        key_col, val_col = cols[0], cols[1]
        for row in reader:
            out[str(row[key_col])] = value_cast(row[val_col])
    return out


def _build_module(args, *, repo_root: Path):
    """Build a Lightning module via the shared config system and load a checkpoint."""
    config_paths = [
        str(repo_root / "inertia1" / "config" / "default.yaml"),
        str(repo_root / "inertia1" / "config" / "methods" / f"{args.method}.yaml"),
    ]
    if args.preset:
        preset_path = repo_root / "inertia1" / "config" / "presets" / f"{args.preset}.yaml"
        if preset_path.exists():
            config_paths.append(str(preset_path))

    overrides = [
        f"data.window_sec={args.window_sec}",
        f"data.hz={args.hz}",
        f"data.axes={args.axes}",
        "run.stage=finetune",
    ]
    cfg = load_configs(config_paths, overrides=overrides)
    preset_name = infer_preset_name_from_paths(config_paths)
    if preset_name is not None:
        cfg["preset_name"] = preset_name
    cfg = apply_preset(cfg)

    from inertia1.data.sensor_config import resolve_data_columns_and_channels

    cfg = resolve_data_columns_and_channels(cfg)
    module = build_experiment(cfg)["module"]

    if args.ckpt_path:
        ckpt = torch.load(str(args.ckpt_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[load] missing={len(missing)} unexpected={len(unexpected)} keys (non-fatal)")
    return module


@torch.no_grad()
def _infer_embed_dim(backbone: nn.Module, *, channels: int, window_samples: int, device: torch.device) -> int:
    backbone.eval()
    dummy = torch.zeros(1, int(channels), int(window_samples), device=device)
    z = backbone.encode(dummy)
    return int(z.reshape(1, -1).shape[1])


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------
def _run_epoch(model, loader, *, kind, device, optimizer=None, class_weight=None):
    train = optimizer is not None
    model.head.train(train)
    model.pool.train(train)
    if kind == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    total_loss, n = 0.0, 0
    all_logits, all_y = [], []
    for bag, y, pos_idx, valid in loader:
        bag = bag.to(device).float()
        y = y.to(device)
        pos_idx = pos_idx.to(device)
        valid = valid.to(device)
        logits = model(bag, pos_idx=pos_idx, valid_mask=valid)
        if kind == "regression":
            loss = loss_fn(logits.squeeze(-1), y.float())
        else:
            loss = loss_fn(logits, y.long())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * y.shape[0]
        n += int(y.shape[0])
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    ys = torch.cat(all_y, dim=0).numpy()
    return total_loss / max(n, 1), logits, ys


def _logits_to_metrics(logits: torch.Tensor, ys: np.ndarray, *, kind: str) -> Dict[str, Any]:
    if kind == "binary":
        prob = torch.softmax(logits, dim=1)[:, 1].numpy() if logits.shape[1] > 1 else torch.sigmoid(logits.squeeze(-1)).numpy()
        return _metrics_binary(ys.astype(int), prob)
    if kind == "multiclass":
        prob = torch.softmax(logits, dim=1).numpy()
        return _metrics_multiclass(ys.astype(int), prob)
    return {"mse": float(np.mean((logits.squeeze(-1).numpy() - ys) ** 2))}


def _class_weights(label_map: Dict[str, Any], ids: List[str], num_classes: int, device) -> torch.Tensor:
    counts = np.zeros(int(num_classes), dtype=np.float64)
    for pid in ids:
        counts[int(label_map[pid])] += 1
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def main() -> None:
    p = argparse.ArgumentParser("inertia1.scripts.eval_nhanes_labels")
    # data
    p.add_argument("--data-root", required=True, help="Dir of per-patient <patient_id>.npy bags [N, C, T].")
    p.add_argument("--labels-csv", required=True, help="CSV: patient_id,label")
    p.add_argument("--split-csv", required=True, help="CSV: patient_id,split (train/val/test)")
    p.add_argument("--task", default="disease", help="Task name (used for output/logging only).")
    p.add_argument("--task-kind", choices=["binary", "multiclass", "regression"], default="binary")
    p.add_argument("--num-classes", type=int, default=2, help="For multiclass; ignored for binary/regression.")
    # backbone
    p.add_argument("--method", choices=["ar_transformer", "patchtst"], default="ar_transformer")
    p.add_argument("--preset", default="small")
    p.add_argument("--ckpt-path", default=None, help="Pretrained checkpoint (last.ckpt). Optional.")
    p.add_argument("--window-sec", type=int, default=30)
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--axes", type=int, default=3)
    # MIL
    p.add_argument("--bag-size", type=int, default=256)
    p.add_argument("--mil-pos-encoding", choices=["learned", "sinusoidal"], default="learned")
    p.add_argument("--mil-hidden-dim", type=int, default=128)
    p.add_argument("--mil-dropout", type=float, default=0.0)
    # optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--class-balanced", action="store_true", help="Use inverse-frequency class weights (classification).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="./logs/disease")
    p.add_argument("--repo-root", default=None, help="Repo root containing inertia1/config (default: inferred).")
    args = p.parse_args()

    _seed_all(args.seed)
    device = _device()
    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kind = args.task_kind
    num_outputs = 1 if kind == "regression" else (2 if kind == "binary" else int(args.num_classes))

    # labels + splits
    cast = float if kind == "regression" else int
    label_map = _read_two_col_csv(args.labels_csv, cast)
    split_map = _read_two_col_csv(args.split_csv, str)
    ids_by_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for pid, split in split_map.items():
        if split in ids_by_split and pid in label_map:
            ids_by_split[split].append(pid)
    for split, ids in ids_by_split.items():
        print(f"[data] {split}: {len(ids)} patients")

    # backbone + dims
    module = _build_module(args, repo_root=repo_root).to(device)
    window_samples = int(round(args.window_sec * args.hz))
    channels = 1 if int(args.axes) == 1 else 3
    embed_dim = _infer_embed_dim(module, channels=channels, window_samples=window_samples, device=device)
    print(f"[model] method={args.method} embed_dim={embed_dim} window_samples={window_samples} channels={channels}")

    model = PatientMILClassifier(
        module,
        embed_dim=embed_dim,
        num_outputs=num_outputs,
        bag_size=args.bag_size,
        mil_hidden=args.mil_hidden_dim,
        mil_dropout=args.mil_dropout,
        pos_encoding=args.mil_pos_encoding,
    ).to(device)

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = PatientWindowBagDataset(
            args.data_root, ids_by_split[split], label_map,
            kind=kind, bag_size=args.bag_size, seed=args.seed,
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=False)

    train_dl = make_loader("train", shuffle=True)
    val_dl = make_loader("val", shuffle=False)
    test_dl = make_loader("test", shuffle=False)

    class_weight = None
    if args.class_balanced and kind in {"binary", "multiclass"}:
        nc = 2 if kind == "binary" else num_outputs
        class_weight = _class_weights(label_map, ids_by_split["train"], nc, device)

    # Only the MIL pooler + head are trainable (backbone is frozen).
    params = list(model.pool.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state, best_test = -float("inf"), None, None
    history = []
    for epoch in range(int(args.epochs)):
        tr_loss, _, _ = _run_epoch(model, train_dl, kind=kind, device=device, optimizer=optimizer, class_weight=class_weight)
        va_loss, va_logits, va_y = _run_epoch(model, val_dl, kind=kind, device=device)
        va_metrics = _logits_to_metrics(va_logits, va_y, kind=kind)
        sel = _select_metric_value(va_metrics, kind) if kind != "regression" else -va_loss
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_select": sel})
        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_select={sel:.4f}")
        if sel > best_val:
            best_val = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_logits, te_y = _run_epoch(model, test_dl, kind=kind, device=device)
    best_test = _logits_to_metrics(te_logits, te_y, kind=kind)
    best_test["test_loss"] = te_loss

    result = {
        "task": args.task,
        "task_kind": kind,
        "method": args.method,
        "preset": args.preset,
        "ckpt_path": args.ckpt_path,
        "bag_size": args.bag_size,
        "best_val_select": best_val,
        "test_metrics": best_test,
        "history": history,
    }
    out_path = out_dir / "nhanes_label_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] test metrics: {json.dumps(best_test, indent=2)}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
