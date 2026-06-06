from __future__ import annotations

import argparse
from omegaconf import OmegaConf

from scripts.train_fda_teacher import _plain, train_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a target-specific LoRA teacher from active-molecule SMILES.")
    parser.add_argument("--config")
    parser.add_argument("--base_model_name_or_path")
    parser.add_argument("--target_data_path")
    parser.add_argument("--target_name", default="target1")
    parser.add_argument("--output_adapter_dir")
    args = parser.parse_args()
    cfg = _plain(OmegaConf.load(args.config)) if args.config else {}
    if args.base_model_name_or_path:
        cfg["base_model_name_or_path"] = args.base_model_name_or_path
    if args.target_data_path:
        cfg["data_path"] = args.target_data_path
    if args.output_adapter_dir:
        cfg["output_adapter_dir"] = args.output_adapter_dir
    cfg["run_name"] = cfg.get("run_name", f"{args.target_name}_teacher_lora")
    train_teacher(cfg, target_name=args.target_name)


if __name__ == "__main__":
    main()
