from __future__ import annotations

from typing import Any, Dict

from inertia1.methods.common import build_datamodule
from inertia1.methods.ar_transformer_module import ARTransformerModule


def build_ar_transformer_experiment(cfg: Dict[str, Any], trainer) -> Dict[str, Any]:
    datamodule = build_datamodule(cfg)
    module = ARTransformerModule(cfg)
    return {"trainer": trainer, "module": module, "datamodule": datamodule}
