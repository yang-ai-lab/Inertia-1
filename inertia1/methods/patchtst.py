from __future__ import annotations

from typing import Any, Dict

from inertia1.methods.common import build_datamodule
from inertia1.methods.patchtst_module import PatchTSTModule


def build_patchtst_experiment(cfg: Dict[str, Any], trainer) -> Dict[str, Any]:
    datamodule = build_datamodule(cfg)
    module = PatchTSTModule(cfg)
    return {"trainer": trainer, "module": module, "datamodule": datamodule}
