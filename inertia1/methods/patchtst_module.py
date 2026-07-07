from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from inertia1.models.backbones.patchtst.patchTST_accel import PatchTST_Accel
from inertia1.augmentations.patchtst import create_patch, random_masking


def masked_patch_mse(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE over patches.

    preds/target: [B, num_patch, n_vars, patch_len]
    mask:         [B, num_patch, n_vars] with 1=masked, 0=kept
    """
    loss = (preds - target) ** 2
    loss = loss.mean(dim=-1)  # [B, num_patch, n_vars]
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class PatchTSTModule(pl.LightningModule):
    """Masked reconstruction pretraining using PatchTST_Accel.

    Canonical batch from UnifiedDataModule: x is Tensor [B, C, T].

    Note: PatchTST_Accel backbone is built for 3-channel accel vectors (x/y/z).
    If your data has C != 3, we adapt channels -> 3 via a learnable 1x1 Conv1d
    (per-timestep linear projection), so the overall pipeline still supports
    arbitrary C while keeping the original PatchTST_Accel definition intact.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.get("method", {})
        params = mcfg.get("params", mcfg)
        optim = cfg.get("optim", {})
        dcfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {}).get("params", {})

        channels = int(dcfg.get("channels", dcfg.get("axes", 3)))
        window_sec = int(dcfg.get("window_sec", 60))
        hz = int(dcfg.get("hz", 20))
        T = window_sec * hz

        # PatchTST params
        self.patch_len = int(params.get("patch_len", 16))
        self.stride = int(params.get("stride", self.patch_len))
        self.mask_ratio = float(params.get("mask_ratio", 0.4))

        # number of patches along time for target length T
        if T < self.patch_len:
            raise ValueError(f"window length T={T} must be >= patch_len={self.patch_len}")
        self.num_patch = 1 + (T - self.patch_len) // self.stride

        self.lr = float(optim.get("lr", 1e-3))
        self.weight_decay = float(optim.get("weight_decay", 1e-4))
        self.beta1 = float(optim.get("beta1", 0.9))
        self.beta2 = float(optim.get("beta2", 0.95))

        # LR scheduler config
        sched_cfg = optim.get("scheduler", {}) or {}
        self.use_scheduler = bool(sched_cfg.get("enabled", True))
        self.warmup_steps = int(sched_cfg.get("warmup_steps", 500))
        self.min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

        # Channel adapter to 3-ch accel if needed
        self.in_channels = channels
        self.backbone_channels = 3
        self.channel_adapter: Optional[nn.Module]
        if channels == 3:
            self.channel_adapter = None
        else:
            self.channel_adapter = nn.Conv1d(channels, 3, kernel_size=1, bias=True)

        # Construct PatchTST_Accel with correct signature ordering
        self.model = PatchTST_Accel(
            patch_len=self.patch_len,
            stride=self.stride,
            num_patch=self.num_patch,
            n_layers=int(model_cfg.get("n_layers", 3)),
            d_model=int(model_cfg.get("d_model", 128)),
            n_heads=int(model_cfg.get("n_heads", 16)),
            d_ff=int(model_cfg.get("d_ff", 256)),
            norm=str(model_cfg.get("norm", "LayerNorm")),
            attn_dropout=float(model_cfg.get("attn_dropout", 0.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            act=str(model_cfg.get("act", "gelu")),
            res_attention=bool(model_cfg.get("res_attention", True)),
            pre_norm=bool(model_cfg.get("pre_norm", False)),
            store_attn=bool(model_cfg.get("store_attn", False)),
            pe=str(model_cfg.get("pe", "zeros")),
            learn_pe=bool(model_cfg.get("learn_pe", True)),
            head_dropout=float(model_cfg.get("head_dropout", 0.0)),
            head_type=str(model_cfg.get("head_type", "pretrain")),
            target_dim=model_cfg.get("target_dim", None),
            verbose=bool(model_cfg.get("verbose", False)),
        )

        # Initialize any lazy layers (e.g., channel_adapter) and validate forward shapes
        with torch.no_grad():
            dummy = torch.randn(2, channels, T)
            if self.channel_adapter is not None:
                dummy = self.channel_adapter(dummy)
            xb = dummy.permute(0, 2, 1)  # [B,T,3]
            xb_patch, _ = create_patch(xb, self.patch_len, self.stride)
            xb_masked, _, _, _ = random_masking(xb_patch, self.mask_ratio)
            _ = self.model(xb_masked)

    def _prep_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self._prep_input(x)  # [B,3,T]
        xb = x.permute(0, 2, 1)  # [B, T, 3]
        xb_patch, _ = create_patch(xb, self.patch_len, self.stride)  # [B, num_patch, 3, patch_len]
        xb_masked, _, mask, _ = random_masking(xb_patch, self.mask_ratio)
        preds = self.model(xb_masked)
        loss = masked_patch_mse(preds, xb_patch, mask)
        return loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean pooled PatchTST backbone representation [B, D]."""
        # x: [B, C, T]
        x = self._prep_input(x)          # [B,3,T]
        xb = x.permute(0, 2, 1)          # [B, T, 3]
        xb_patch, _ = create_patch(xb, self.patch_len, self.stride)  # [B, num_patch, 3, patch_len]
        z = self.model.backbone(xb_patch)  # [B, num_patch, d_model]
        return z.mean(dim=1)             # [B, d_model]

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
        )

        if not self.use_scheduler:
            return opt

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.warmup_steps
        min_lr_ratio = self.min_lr_ratio

        def lr_lambda(current_step: int) -> float:
            # Linear warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay to min_lr_ratio after warmup
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
