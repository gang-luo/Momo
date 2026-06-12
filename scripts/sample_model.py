from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from molopd.utils.smiles import canonicalize_smiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample base, LoRA-teacher, or MolOPD student SMILES.")
    parser.add_argument("--model_name_or_path", required=True, help="Base HF model or MolOPD full model path.")
    parser.add_argument("--tokenizer_name_or_path")
    parser.add_argument("--adapter_path", help="FDA/target/MolOPD LoRA adapter path to attach to the base model.")
    parser.add_argument("--label", default="model")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path or args.adapter_path or args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    encoded = tokenizer([args.prompt], return_tensors="pt")
    out = model.generate(
        **encoded,
        do_sample=True,
        num_return_sequences=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.batch_decode(out, skip_special_tokens=True)
    rows = []
    for s in raw:
        canon = canonicalize_smiles(s)
        rows.append({"smiles": canon or "", "raw_smiles": s, "valid": canon is not None, "label": args.label, "model_name": args.label})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["smiles", "raw_smiles", "valid", "label", "model_name"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
