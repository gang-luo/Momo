from __future__ import annotations


class AdmetAiEvaluator:
    """Placeholder ADMET-AI adapter; disabled by default in MolOPD v1."""

    def __init__(self, enabled: bool = False, **_: object):
        self.enabled = enabled
        if enabled:
            raise NotImplementedError("ADMET-AI evaluator is an optional interface and is not wired into MolOPD v1 core loss.")

    def __call__(self, smiles: list[str]) -> list[dict[str, float]]:
        return [{} for _ in smiles]
