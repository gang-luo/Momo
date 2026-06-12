from __future__ import annotations

from pathlib import Path

from molopd.data.datamodule import MolDataModule
from molopd.training.config import (
    build_checkpoint_callbacks,
    build_logger,
    parse_config_only,
    pl,
    trainer_kwargs,
)
from molopd.training.teacher_lightning_module import MolTeacherSFTLightningModule


def run_teacher_training(cfg: dict, *, default_target_name: str) -> None:
    runtime = cfg.get("runtime", {})
    logging_cfg = cfg.get("logging", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    teacher_cfg = cfg.get("teacher", {})

    target_name = teacher_cfg.get("target_name", default_target_name)
    output_dir = Path(runtime.get("output_dir", logging_cfg.get("output_dir", f"outputs/{target_name}_teacher")))
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(int(runtime.get("seed", 56)), workers=True)

    dm = MolDataModule(
        model_name=model_cfg.get("tokenizer_name_or_path") or model_cfg["base_model_name_or_path"],
        train_path=data_cfg["train_path"],
        val_path=data_cfg.get("val_path"),
        test_path=data_cfg.get("test_path"),
        batch_size=int(data_cfg.get("batch_size", 8)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        max_length=int(data_cfg.get("max_length", 256)),
        sample_size=data_cfg.get("sample_size"),
    )

    lit_model = MolTeacherSFTLightningModule(
        model_cfg=model_cfg,
        lora_cfg=cfg.get("lora", {}),
        optimizer_cfg=cfg.get("optimizer", {}),
        scheduler_cfg=cfg.get("scheduler", {}),
        loss_cfg=cfg.get("loss", {}),
        teacher_cfg=teacher_cfg,
    )

    logger = build_logger(logging_cfg, output_dir)
    callbacks = build_checkpoint_callbacks(output_dir, cfg.get("checkpoint", {}), default_monitor="val/loss" if data_cfg.get("val_path") else None)
    trainer = pl.Trainer(**trainer_kwargs(cfg, output_dir, logger, callbacks))
    ckpt_dir = output_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if bool(runtime.get("resume", False)) and last_ckpt.exists() else None
    trainer.fit(lit_model, datamodule=dm, ckpt_path=ckpt_path)
    lit_model.save_adapter(model_cfg["output_adapter_dir"])
    if bool(runtime.get("run_test", False)) and data_cfg.get("test_path"):
        trainer.test(lit_model, datamodule=dm, ckpt_path="best" if data_cfg.get("val_path") else ckpt_path)


def main() -> None:
    cfg = parse_config_only("Train the FDA LoRA teacher with Lightning. All parameters come from YAML.")
    run_teacher_training(cfg, default_target_name="fda")


if __name__ == "__main__":
    main()
