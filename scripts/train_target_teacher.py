from __future__ import annotations

from molopd.training.config import parse_config_only
from scripts.train_fda_teacher import run_teacher_training


def main() -> None:
    cfg = parse_config_only("Train a target-specific LoRA teacher with Lightning. All parameters come from YAML.")
    run_teacher_training(cfg, default_target_name=cfg.get("teacher", {}).get("target_name", "target1"))


if __name__ == "__main__":
    main()
