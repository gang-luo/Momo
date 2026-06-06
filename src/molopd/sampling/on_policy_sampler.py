from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from molopd.utils.smiles import canonicalize_smiles


@dataclass
class OnPolicySampleBatch:
    raw_smiles: list[str]
    smiles: list[str]
    valid: list[bool]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    valid_ratio: float
    unique_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class OnPolicySampler:
    """Samples from the current student and prepares valid canonical SMILES for OPD."""

    def __init__(self, student, tokenizer: PreTrainedTokenizerBase | None = None, *, prompt: str = ""):
        self.student = student
        self.tokenizer = tokenizer or student.tokenizer
        self.prompt = prompt

    def sample(
        self,
        *,
        num_samples: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        keep_duplicates: bool = False,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        raw = self.student.sample(
            self.prompt,
            num_return_sequences=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        canonical_all: list[str | None] = [canonicalize_smiles(s) for s in raw]
        valid_flags = [s is not None for s in canonical_all]
        valid_smiles = [s for s in canonical_all if s is not None]
        if keep_duplicates:
            smiles = valid_smiles
        else:
            seen: set[str] = set()
            smiles = []
            for s in valid_smiles:
                if s not in seen:
                    smiles.append(s)
                    seen.add(s)

        # Keep the loss path robust even if a sampling step returns no valid molecule.
        if not smiles:
            smiles = ["C"]

        tokens = self.tokenizer(
            smiles,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        labels[tokens["attention_mask"] == 0] = -100
        valid_ratio = sum(valid_flags) / max(len(valid_flags), 1)
        unique_ratio = len(set(valid_smiles)) / max(len(valid_smiles), 1) if valid_smiles else 0.0
        return OnPolicySampleBatch(
            raw_smiles=raw,
            smiles=smiles,
            valid=valid_flags,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            labels=labels,
            valid_ratio=float(valid_ratio),
            unique_ratio=float(unique_ratio),
        ).as_dict()
