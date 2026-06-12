from __future__ import annotations


class TargetEvaluator:
    """Placeholder target scorer interface; logging/reward integration is future work."""

    def __init__(self, enabled: bool = False, **_: object):
        self.enabled = enabled
        if enabled:
            raise NotImplementedError("Target evaluator is not part of MolOPD v1 token-level loss.")

    def __call__(self, smiles: list[str]) -> list[float | None]:
        return [None for _ in smiles]
