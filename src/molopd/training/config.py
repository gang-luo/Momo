from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
    from pytorch_lightning.strategies import DDPStrategy


def to_plain(obj: Any) -> Any:
    if hasattr(obj, "items"):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = to_plain(OmegaConf.load(path))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    cfg["config_path"] = str(path)
    return cfg


def parse_config_only(description: str) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to the YAML config. All run parameters are read from this file.")
    args = parser.parse_args()
    return load_config(args.config)


def resolve_trainer_strategy(trainer_cfg: dict[str, Any]) -> Any:
    strategy = str(trainer_cfg.get("strategy", "auto")).strip().lower()
    if strategy in {"", "auto", "none"}:
        return "auto"
    if strategy == "ddp":
        return DDPStrategy(
            find_unused_parameters=bool(trainer_cfg.get("ddp_find_unused_parameters", True)),
            static_graph=bool(trainer_cfg.get("ddp_static_graph", False)),
            gradient_as_bucket_view=True,
        )
    return trainer_cfg.get("strategy", "auto")


def build_logger(logging_cfg: dict[str, Any], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(logging_cfg.get("use_wandb", False)) and not bool(logging_cfg.get("disabled", False)):
        if logging_cfg.get("api_key"):
            os.environ["WANDB_API_KEY"] = str(logging_cfg["api_key"])
        return WandbLogger(
            project=logging_cfg.get("project", "MolOPD"),
            name=logging_cfg.get("run_name", "molopd"),
            entity=logging_cfg.get("entity") or None,
            save_dir=str(output_dir),
            log_model=bool(logging_cfg.get("log_model", False)),
        )
    return CSVLogger(str(output_dir), name="csv_logs")


def build_checkpoint_callbacks(
    output_dir: str | Path,
    checkpoint_cfg: dict[str, Any] | None = None,
    *,
    default_monitor: str | None = None,
) -> list[Any]:
    checkpoint_cfg = checkpoint_cfg or {}
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = checkpoint_cfg.get("monitor", default_monitor)
    kwargs: dict[str, Any] = {
        "dirpath": str(ckpt_dir),
        "filename": checkpoint_cfg.get("filename", "{epoch:02d}-{step}"),
        "save_last": bool(checkpoint_cfg.get("save_last", True)),
        "save_top_k": int(checkpoint_cfg.get("save_top_k", 1 if monitor else -1)),
    }
    if monitor:
        kwargs["monitor"] = monitor
        kwargs["mode"] = checkpoint_cfg.get("mode", "min")
    if checkpoint_cfg.get("every_n_train_steps") is not None:
        kwargs["every_n_train_steps"] = int(checkpoint_cfg["every_n_train_steps"])
    callbacks = [ModelCheckpoint(**kwargs)]
    if bool(checkpoint_cfg.get("log_lr", True)):
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def trainer_kwargs(cfg: dict[str, Any], output_dir: str | Path, logger: Any, callbacks: list[Any]) -> dict[str, Any]:
    trainer = cfg.get("trainer", {})
    kwargs = {
        "default_root_dir": str(output_dir),
        "max_epochs": int(trainer.get("max_epochs", 1)),
        "accelerator": trainer.get("accelerator", "auto"),
        "devices": trainer.get("devices", "auto"),
        "precision": trainer.get("precision", "32-true"),
        "strategy": resolve_trainer_strategy(trainer),
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": int(trainer.get("log_every_n_steps", 10)),
        "accumulate_grad_batches": max(1, int(trainer.get("accumulate_grad_batches", 1))),
        "num_sanity_val_steps": max(0, int(trainer.get("num_sanity_val_steps", 0))),
        "gradient_clip_val": float(trainer.get("gradient_clip_val", cfg.get("optimizer", {}).get("grad_clip", 0.0))),
    }
    if trainer.get("max_steps") is not None:
        kwargs["max_steps"] = int(trainer["max_steps"])
    if trainer.get("check_val_every_n_epoch") is not None:
        kwargs["check_val_every_n_epoch"] = int(trainer["check_val_every_n_epoch"])
    if trainer.get("val_check_interval") is not None:
        kwargs["val_check_interval"] = trainer["val_check_interval"]
    return kwargs


__all__ = [
    "pl",
    "load_config",
    "parse_config_only",
    "build_logger",
    "build_checkpoint_callbacks",
    "trainer_kwargs",
    "to_plain",
]
