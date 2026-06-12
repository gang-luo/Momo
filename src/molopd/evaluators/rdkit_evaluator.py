from __future__ import annotations

from molopd.metrics.mol_metrics import compute_basic_generation_metrics


class RdkitEvaluator:
    def __call__(self, smiles: list[str]) -> dict[str, float]:
        return compute_basic_generation_metrics(smiles)
