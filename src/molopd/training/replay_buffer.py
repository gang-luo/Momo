from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvReplayBuffer:
    """Append-only CSVL-style sample log for MolOPD v1."""

    DEFAULT_FIELDS = [
        "iteration",
        "raw_smiles",
        "canonical_smiles",
        "valid",
        "logp_student",
        "logp_base",
        "logp_fda_teacher",
        "task_score",
        "admet_scores",
        "fda_likeness",
        "novelty",
        "diversity",
        "total_reward",
        "scaffold",
    ]

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replay_path = self.output_dir / "replay_buffer.csv"

    def write_iteration(self, iteration: int, rows: list[dict[str, Any]]) -> Path:
        iter_path = self.output_dir / f"samples_iter_{iteration:04d}.csv"
        fields = self._fields(rows)
        self._write_csv(iter_path, rows, fields, append=False)
        self._write_csv(self.replay_path, rows, fields, append=self.replay_path.exists())
        return iter_path

    def _fields(self, rows: list[dict[str, Any]]) -> list[str]:
        keys = set().union(*(r.keys() for r in rows)) if rows else set()
        fields = list(self.DEFAULT_FIELDS)
        for key in sorted(keys):
            if key not in fields:
                fields.append(key)
        return fields

    def _write_csv(self, path: Path, rows: list[dict[str, Any]], fields: list[str], *, append: bool) -> None:
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not append:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
