from __future__ import annotations

import argparse
import os
from typing import Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from molopd.opd.molopd_lightning_module import MolOPDLightningModule


def _plain(x: Any) -> Any:
    if hasattr(x, "items"):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Run on-policy token-level multi-teacher MolOPD training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = _plain(OmegaConf.load(args.config))
    pl.seed_everything(int(cfg.get("runtime", {}).get("seed", cfg.get("training", {}).get("seed", 56))), workers=True)
    logging_cfg = cfg.get("logging", {})
    logger = None
    if bool(logging_cfg.get("use_wandb", False)):
        if logging_cfg.get("api_key"):
            os.environ["WANDB_API_KEY"] = str(logging_cfg["api_key"])
        logger = WandbLogger(project=logging_cfg.get("project", "MolOPD"), name=logging_cfg.get("run_name", "molopd"), entity=logging_cfg.get("entity"))
    model = MolOPDLightningModule(cfg)
    train_cfg = cfg.get("training", {})
    ckpt = ModelCheckpoint(
        dirpath=f"{logging_cfg.get('output_dir', 'outputs/molopd')}/{logging_cfg.get('run_name', 'run')}/checkpoints",
        filename="{step}-molopd-{train/loss_total:.4f}",
        save_last=True,
        every_n_train_steps=int(train_cfg.get("save_every_n_steps", 500)),
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_steps=int(train_cfg.get("max_steps", 10000)),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", "auto"),
        precision=train_cfg.get("precision", "bf16-mixed"),
        logger=logger,
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 10)),
        accumulate_grad_batches=int(train_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 1.0)),
        callbacks=[ckpt],
        default_root_dir=logging_cfg.get("output_dir", "outputs/molopd"),
        enable_checkpointing=True,
    )
    trainer.fit(model)
    model.student.save_pretrained(f"{logging_cfg.get('output_dir', 'outputs/molopd')}/{logging_cfg.get('run_name', 'run')}/final_student")


if __name__ == "__main__":
    main()
