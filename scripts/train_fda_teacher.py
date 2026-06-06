from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from molopd.data.smiles_dataset import SmilesDataset


def _plain(x: Any) -> Any:
    if hasattr(x, "items"):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def _collator(ds: SmilesDataset):
    def collate(batch):
        out = ds.collate_fn(batch)
        return {k: v for k, v in out.items() if k not in {"smiles", "raw_smiles"}}
    return collate


def train_teacher(cfg: dict[str, Any], *, target_name: str = "fda") -> None:
    set_seed(int(cfg.get("seed", 56)))
    base_model = cfg["base_model_name_or_path"]
    tokenizer_path = cfg.get("tokenizer_name_or_path") or base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=bool(cfg.get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=bool(cfg.get("trust_remote_code", False)))
    lora = cfg.get("lora", {})
    model = get_peft_model(model, LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get("target_modules"),
        bias=lora.get("bias", "none"),
        task_type="CAUSAL_LM",
    ))
    train_ds = SmilesDataset(cfg["data_path"], tokenizer, max_length=int(cfg.get("max_length", 256)), sample_size=cfg.get("sample_size"))
    eval_ds = SmilesDataset(cfg["val_data_path"], tokenizer, max_length=int(cfg.get("max_length", 256))) if cfg.get("val_data_path") else None
    output_dir = Path(cfg["output_adapter_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg.get("training", {})
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", tcfg.get("batch_size", 8))),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", tcfg.get("batch_size", 8))),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(tcfg.get("lr", 2e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 3)),
        max_steps=int(tcfg.get("max_steps", -1)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 500)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(tcfg.get("eval_steps", 500)),
        bf16=bool(tcfg.get("bf16", False)),
        fp16=bool(tcfg.get("fp16", False)),
        report_to=["wandb"] if bool(cfg.get("use_wandb", False)) else [],
        run_name=cfg.get("run_name", f"{target_name}_teacher_lora"),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=_collator(train_ds))
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "teacher_config.json").write_text(json.dumps({"target_name": target_name, **cfg}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an FDA LoRA teacher from approved-drug SMILES.")
    parser.add_argument("--config")
    parser.add_argument("--base_model_name_or_path")
    parser.add_argument("--fda_data_path")
    parser.add_argument("--output_adapter_dir")
    args = parser.parse_args()
    cfg = _plain(OmegaConf.load(args.config)) if args.config else {}
    if args.base_model_name_or_path:
        cfg["base_model_name_or_path"] = args.base_model_name_or_path
    if args.fda_data_path:
        cfg["data_path"] = args.fda_data_path
    if args.output_adapter_dir:
        cfg["output_adapter_dir"] = args.output_adapter_dir
    train_teacher(cfg, target_name="fda")


if __name__ == "__main__":
    main()
