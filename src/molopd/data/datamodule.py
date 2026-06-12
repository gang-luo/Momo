from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from molopd.data.smiles_dataset import SmilesDataset, load_smiles_records


class JsonlSmilesDataset(SmilesDataset):
    """Backward-compatible alias; now supports JSON/JSONL/CSV and canonicalization."""

    def __init__(self, path: str | Path, tokenizer: Any | None = None):
        if tokenizer is None:
            # Raw-record mode for legacy callers that only index rows.
            self.records = load_smiles_records(path, canonicalize=False, drop_invalid=False, deduplicate=False)
            self.tokenizer = None
            self.max_length = 256
        else:
            super().__init__(path, tokenizer)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


class MolDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        train_path: str,
        val_path: str | None = None,
        test_path: str | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 256,
        sample_size: int | None = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.sample_size = sample_size

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = SmilesDataset(self.train_path, self.tokenizer, max_length=self.max_length, sample_size=self.sample_size)
        self.val_ds = SmilesDataset(self.val_path, self.tokenizer, max_length=self.max_length) if self.val_path else None
        self.test_ds = SmilesDataset(self.test_path, self.tokenizer, max_length=self.max_length) if self.test_path else None

    @staticmethod
    def _strip_metadata(batch: dict[str, Any], *, keep_metadata: bool = True):
        if keep_metadata:
            return batch
        return {k: v for k, v in batch.items() if k not in {"smiles", "raw_smiles"}}

    def _collate_train(self, batch: list[dict[str, Any]]):
        return self.train_ds.collate_fn(batch)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate_train)

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.val_ds.collate_fn)

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.test_ds.collate_fn)
