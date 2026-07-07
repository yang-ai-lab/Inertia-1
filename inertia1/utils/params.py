from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn


def count_parameters(module: nn.Module, cfg: Optional[Dict[str, Any]] = None) -> Tuple[int, int, Dict[str, int]]:
    """Count parameters (optionally initializing lazy modules via a dummy forward).

    Args:
      module: a torch module (can be a LightningModule too).
      cfg: experiment config dict. If provided, we will run a dummy forward
           with shape (64, channels, window_sec * hz) to initialize Lazy* layers.

    Returns:
      total_params, trainable_params, breakdown
    Breakdown is by top-level child module name.
    """
    if cfg is not None:
        try:
            module_was_training = module.training
            module.eval()
            with torch.no_grad():
                channels = cfg.get("data", {}).get("axes", 3)
                window_sec = cfg.get("data", {}).get("window_sec", 60)
                hz = cfg.get("data", {}).get("hz", 20)
                dummy_input = torch.randn(64, int(channels), int(window_sec * hz))
                # Try to place dummy on the module's device, if any params exist
                try:
                    device = next(module.parameters()).device
                    dummy_input = dummy_input.to(device)
                except StopIteration:
                    pass

                # Prefer module.forward(x). If it's a LightningModule that expects training_step batches,
                # it should still usually implement forward(x). If not, we best-effort call it.
                module(dummy_input)
        except Exception:
            # Best-effort initialization only; parameter counting should never crash training.
            pass
        finally:
            try:
                module.train(module_was_training)  # type: ignore
            except Exception:
                pass

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

    breakdown: Dict[str, int] = OrderedDict()
    for name, child in module.named_children():
        breakdown[name] = sum(p.numel() for p in child.parameters())
    if not breakdown:
        breakdown["(all)"] = total
    return total, trainable, breakdown
