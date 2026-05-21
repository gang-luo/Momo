from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class JsonlSmilesDataset(Dataset):
    def __init__(self, path: str | Path):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.items.append(row)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


@dataclass
class BatchConfig:
    max_length: int = 256


class MolDataModule(pl.LightningDataModule):
    def __init__(self, model_name: str, train_path: str, val_path: str, test_path: str, batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = JsonlSmilesDataset(self.train_path)
        self.val_ds = JsonlSmilesDataset(self.val_path)
        self.test_ds = JsonlSmilesDataset(self.test_path)

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        smiles = [x["smiles"] for x in batch]
        tokens = self.tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate)
