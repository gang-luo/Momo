from __future__ import annotations

from pathlib import Path

from molopd.opd.molopd_lightning_module import MolOPDLightningModule
from molopd.training.config import (
    build_checkpoint_callbacks,
    build_logger,
    parse_config_only,
    pl,
    trainer_kwargs,
)


def main() -> None:
    cfg = parse_config_only("Run MolOPD token-level multi-teacher training. All parameters come from YAML.")
    runtime = cfg.get("runtime", {})
    logging_cfg = cfg.get("logging", {})
    output_dir = Path(runtime.get("output_dir", logging_cfg.get("output_dir", "outputs/molopd")))
    run_name = logging_cfg.get("run_name", "molopd")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(runtime.get("seed", 56)), workers=True)
    logger = build_logger(logging_cfg, run_dir)
    callbacks = build_checkpoint_callbacks(run_dir, cfg.get("checkpoint", {}), default_monitor=None)
    model = MolOPDLightningModule(cfg)
    trainer = pl.Trainer(**trainer_kwargs(cfg, run_dir, logger, callbacks))

    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if bool(runtime.get("resume", False)) and last_ckpt.exists() else None
    trainer.fit(model, ckpt_path=ckpt_path)
    model.student.save_pretrained(run_dir / "final_student")


if __name__ == "__main__":
    main()
