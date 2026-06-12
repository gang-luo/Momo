from __future__ import annotations

import json

import pytorch_lightning as pl

from molopd.data.datamodule import MolDataModule
from molopd.training.teacher_lightning_module import MolTeacherSFTLightningModule


def test_teacher_lora_lightning_runs_one_step(tmp_path, tiny_model_dir):
    data_path = tmp_path / "teacher.jsonl"
    data_path.write_text("\n".join(json.dumps({"smiles": s}) for s in ["C", "CCO"]), encoding="utf-8")
    dm = MolDataModule(
        model_name=str(tiny_model_dir),
        train_path=str(data_path),
        val_path=str(data_path),
        batch_size=1,
        num_workers=0,
        max_length=16,
    )
    lit = MolTeacherSFTLightningModule(
        model_cfg={"base_model_name_or_path": str(tiny_model_dir), "tokenizer_name_or_path": str(tiny_model_dir), "output_adapter_dir": str(tmp_path / "adapter")},
        lora_cfg={"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["c_attn"]},
        optimizer_cfg={"lr": 1e-4, "weight_decay": 0.0},
        scheduler_cfg={"name": "none"},
        loss_cfg={"label_pad_token_id": -100},
        teacher_cfg={"target_name": "fda"},
    )
    trainer = pl.Trainer(max_steps=1, accelerator="cpu", devices=1, precision="32-true", logger=False, enable_checkpointing=False)
    trainer.fit(lit, datamodule=dm)
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
