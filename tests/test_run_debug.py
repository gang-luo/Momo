from __future__ import annotations

import pytorch_lightning as pl

from molopd.opd.molopd_lightning_module import MolOPDLightningModule


def test_debug_config_runs_one_step(tmp_path, tiny_model_dir, tiny_adapter_dir):
    cfg = {
        "model": {
            "student_model_name_or_path": str(tiny_model_dir),
            "base_retention_model_name_or_path": str(tiny_model_dir),
            "tokenizer_name_or_path": str(tiny_model_dir),
            "use_lora_for_student": True,
            "student_lora": {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["c_attn"]},
        },
        "teachers": {"fda_adapter_path": str(tiny_adapter_dir), "target_adapters": [{"name": "target1", "adapter_path": str(tiny_adapter_dir), "weight": 1.0}]},
        "loss": {"lambda_fda": 0.5, "lambda_base": 0.1, "lambda_targets": {"target1": 1.0}, "base_kl_direction": "student_to_base", "lambda_nll": 0.0},
        "sampling": {"prompt": "C", "num_samples_per_step": 2, "max_new_tokens": 2, "keep_duplicates": False, "max_length": 16},
        "training": {"lr": 1e-4, "max_steps": 1, "precision": "32-true", "sample_log_every_n_steps": 999},
        "logging": {"output_dir": str(tmp_path), "run_name": "debug", "use_wandb": False},
        "evaluators": {"use_rdkit_eval_for_logging": False},
    }
    model = MolOPDLightningModule(cfg)
    trainer = pl.Trainer(max_steps=1, accelerator="cpu", devices=1, precision="32-true", logger=False, enable_checkpointing=False)
    trainer.fit(model)
