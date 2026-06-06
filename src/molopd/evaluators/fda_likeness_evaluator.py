from __future__ import annotations


class FdaLikenessEvaluator:
    """Placeholder FDA-likeness classifier interface; disabled by default."""

    def __init__(self, enabled: bool = False, **_: object):
        self.enabled = enabled
        if enabled:
            raise NotImplementedError("FDA-likeness classifier is configured as a future optional evaluator.")

    def __call__(self, smiles: list[str]) -> list[float | None]:
        return [None for _ in smiles]
