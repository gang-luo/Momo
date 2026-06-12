from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from molopd.utils.smiles import SMILES_FIELDS, canonicalize_smiles, extract_smiles


def _read_json(path: Path) -> Iterable[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        rows = data.get("data", data.get("items", data.get("molecules")))
        if isinstance(rows, list):
            yield from rows
        else:
            yield data
    else:
        raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data)}")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def load_smiles_records(
    path: str | Path,
    *,
    canonicalize: bool = True,
    drop_invalid: bool = True,
    deduplicate: bool = True,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Load JSON/JSONL/CSV molecule rows and normalize to raw/canonical SMILES records."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif suffix == ".json":
        rows = _read_json(path)
    elif suffix == ".csv":
        rows = _read_csv(path)
    else:
        raise ValueError(f"Unsupported SMILES data format for {path}; expected .json, .jsonl or .csv")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        raw = extract_smiles(row)
        if raw is None:
            continue
        canon = canonicalize_smiles(raw) if canonicalize else raw.strip()
        valid = canon is not None
        if drop_invalid and not valid:
            continue
        smiles = canon if canon is not None else raw.strip()
        if deduplicate and smiles in seen:
            continue
        seen.add(smiles)
        records.append({"raw_smiles": raw, "smiles": smiles, "valid": valid, **row})
        if sample_size is not None and len(records) >= int(sample_size):
            break
    return records


class SmilesDataset(Dataset):
    """Causal-LM dataset for FDA/target LoRA construction and optional warmup."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 256,
        sample_size: int | None = None,
        canonicalize: bool = True,
        deduplicate: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = load_smiles_records(
            path,
            canonicalize=canonicalize,
            drop_invalid=True,
            deduplicate=deduplicate,
            sample_size=sample_size,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
        smiles = [x["smiles"] for x in batch]
        tokens = self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[tokens["attention_mask"] == 0] = -100
        tokens["labels"] = labels
        tokens["smiles"] = smiles
        tokens["raw_smiles"] = [x["raw_smiles"] for x in batch]
        return tokens


__all__ = ["SMILES_FIELDS", "SmilesDataset", "load_smiles_records"]
