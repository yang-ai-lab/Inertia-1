from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class ARTransformerModule(pl.LightningModule):
    """Autoregressive Transformer baseline operating at the patch level.

    Input: x [B, C, T]
    Patches the input into non-overlapping patches of [C * patch_len],
    then predicts next patch from previous patches (MSE) with causal masking.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.get("method", {})
        params = mcfg.get("params", mcfg)
        optim = cfg.get("optim", {})

        dcfg = cfg.get("data", {})
        channels = int(dcfg.get("channels", dcfg.get("axes", 3)))

        self.patch_len = int(params.get("patch_len", 16))
        self.channels = channels
        patch_dim = channels * self.patch_len

        d_model = int(params.get("d_model", 256))
        nhead = int(params.get("n_heads", 8))
        num_layers = int(params.get("n_layers", 2))
        dim_feedforward = int(params.get("d_ff", d_model * 4))
        dropout = float(params.get("dropout", 0.1))

        self.lr = float(optim.get("lr", 1e-3))
        self.weight_decay = float(optim.get("weight_decay", 1e-4))

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_model, patch_dim)
        self.d_model = d_model

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] -> patches: [B, N, C * patch_len]"""
        B, C, T = x.shape
        n_patches = T // self.patch_len
        T_trim = n_patches * self.patch_len
        x = x[:, :, :T_trim]
        x = x.reshape(B, C, n_patches, self.patch_len)
        x = x.permute(0, 2, 1, 3).reshape(B, n_patches, C * self.patch_len)
        return x

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._to_patches(x)  # [B, N, C*patch_len]
        p_in = self.patch_proj(patches[:, :-1, :])  # [B, N-1, d_model]
        h = self.pos_encoder(p_in)

        seq_len = h.size(1)
        causal_mask = self._generate_causal_mask(seq_len).to(h.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)  # [B, N-1, d_model]

        y_hat = self.head(h)  # [B, N-1, C*patch_len]
        return y_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled transformer hidden representation [B, d_model]."""
        patches = self._to_patches(x)  # [B, N, C*patch_len]
        p_in = self.patch_proj(patches)  # [B, N, d_model]
        h = self.pos_encoder(p_in)
        h = self.transformer(h)  # [B, N, d_model]
        return h.mean(dim=1)  # [B, d_model]

    def _step(self, batch, stage: str) -> torch.Tensor:
        x = batch
        patches = self._to_patches(x)  # [B, N, C*patch_len]
        target = patches[:, 1:, :]  # [B, N-1, C*patch_len]
        y_hat = self.forward(x)  # [B, N-1, C*patch_len]
        loss = torch.mean((y_hat - target) ** 2)
        self.log(f"{stage}/mse", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        ocfg = self.cfg.get("optim", {})
        lr = float(ocfg.get("lr", self.lr))
        wd = float(ocfg.get("weight_decay", self.weight_decay))
        betas = ocfg.get("betas", (0.9, 0.999))
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=tuple(betas))

        scfg = ocfg.get("scheduler", {})
        if scfg.get("enabled", True):
            warmup_steps = int(scfg.get("warmup_steps", 500))
            min_lr_ratio = float(scfg.get("min_lr_ratio", 0.01))
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                
                progress = float(current_step - warmup_steps) / float(max(1, self.trainer.estimated_stepping_batches - warmup_steps))
                progress = min(1.0, max(0.0, progress))
                
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        
        return optimizer
