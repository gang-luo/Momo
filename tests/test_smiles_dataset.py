from __future__ import annotations

import json

from transformers import AutoTokenizer

from molopd.data.smiles_dataset import SmilesDataset, load_smiles_records


def test_smiles_dataset_reads_canonicalizes_deduplicates_and_tokenizes(tmp_path, tiny_model_dir):
    path = tmp_path / "mols.jsonl"
    rows = [{"SMILES": "C(C)O"}, {"smiles": "CCO"}, {"canonical_smiles": "not_a_smiles"}, {"generated_smiles": "N"}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    records = load_smiles_records(path)
    assert [r["smiles"] for r in records] == ["CCO", "N"]
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    ds = SmilesDataset(path, tok)
    batch = ds.collate_fn([ds[0], ds[1]])
    assert set(["input_ids", "attention_mask", "labels", "smiles", "raw_smiles"]).issubset(batch)
    assert batch["input_ids"].shape == batch["labels"].shape
