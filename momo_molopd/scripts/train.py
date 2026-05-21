from __future__ import annotations

import argparse
import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from molopd.data.datamodule import MolDataModule
from molopd.models.lightning_module import MolOPDLightningModule, MolTeacherLightningModule


def _build_logger(cfg):
    wandb_cfg = cfg.wandb
    if wandb_cfg.disabled:
        return None
    if wandb_cfg.api_key:
        os.environ["WANDB_API_KEY"] = str(wandb_cfg.api_key)
    return WandbLogger(project=wandb_cfg.project, name=wandb_cfg.run_name, entity=wandb_cfg.entity)


def _build_strategy(cfg):
    if str(cfg.trainer.strategy).lower() == "ddp":
        return DDPStrategy(
            static_graph=bool(cfg.trainer.ddp_static_graph),
            find_unused_parameters=bool(cfg.trainer.ddp_find_unused_parameters),
        )
    return cfg.trainer.strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    pl.seed_everything(int(cfg.runtime.seed), workers=True)

    dm = MolDataModule(**cfg.data)

    if cfg.task == "teacher":
        model = MolTeacherLightningModule(**cfg.model)
    elif cfg.task == "opd":
        model = MolOPDLightningModule(**cfg.model)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    logger = _build_logger(cfg)
    monitor_metric = str(cfg.trainer.monitor)
    monitor_tag = monitor_metric.replace("/", "_")
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(cfg.runtime.output_dir, "checkpoints"),
        filename=f"{{epoch:02d}}-{{step}}-{monitor_tag}" + "={" + monitor_metric + ":.4f}",
        save_top_k=3,
        save_last=True,
        monitor=monitor_metric,
        mode=cfg.trainer.monitor_mode,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=_build_strategy(cfg),
        precision=cfg.trainer.precision,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        default_root_dir=cfg.runtime.output_dir,
        callbacks=[checkpoint_cb],
    )

    ckpt_path = "last" if bool(cfg.runtime.resume) else None
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    if bool(cfg.runtime.run_test):
        trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
