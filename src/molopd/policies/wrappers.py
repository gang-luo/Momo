from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def assert_tokenizer_compatible(student_tokenizer: PreTrainedTokenizerBase, other_tokenizer: PreTrainedTokenizerBase, name: str) -> None:
    if len(student_tokenizer) != len(other_tokenizer):
        raise ValueError(
            f"Tokenizer mismatch for {name}: student vocab={len(student_tokenizer)} vs {name} vocab={len(other_tokenizer)}. "
            "MolOPD v1 does not support tokenizer mismatch. Please use a shared tokenizer_name_or_path."
        )
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        if getattr(student_tokenizer, attr, None) != getattr(other_tokenizer, attr, None):
            raise ValueError(
                f"Tokenizer mismatch for {name}: {attr} differs. MolOPD v1 does not support tokenizer mismatch."
            )


@dataclass
class PolicyLoadConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    torch_dtype: str | None = None
    trust_remote_code: bool = False
    device_map: str | dict[str, Any] | None = None

    def dtype(self):
        if self.torch_dtype is None or self.torch_dtype == "auto":
            return "auto" if self.torch_dtype == "auto" else None
        return getattr(torch, self.torch_dtype)


class BasePolicyWrapper(torch.nn.Module):
    """Frozen wrapper around a Hugging Face autoregressive SMILES generator."""

    def __init__(self, model_name_or_path: str, tokenizer_name_or_path: str | None = None, *, frozen: bool = True, **kwargs: Any):
        super().__init__()
        self.load_config = PolicyLoadConfig(model_name_or_path, tokenizer_name_or_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path, trust_remote_code=self.load_config.trust_remote_code)
        _ensure_pad_token(self.tokenizer)
        model_kwargs: dict[str, Any] = {"trust_remote_code": self.load_config.trust_remote_code}
        if self.load_config.dtype() is not None:
            model_kwargs["torch_dtype"] = self.load_config.dtype()
        if self.load_config.device_map is not None:
            model_kwargs["device_map"] = self.load_config.device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if frozen:
            _freeze(self.model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, **batch: torch.Tensor):
        return self.model(**batch)

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    @torch.no_grad()
    def sample(
        self,
        prompts: str | list[str] = "",
        *,
        num_return_sequences: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **encoded,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def token_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.logits(input_ids=input_ids, attention_mask=attention_mask)
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        target = input_ids[:, 1:].unsqueeze(-1)
        token_logp = logp.gather(-1, target).squeeze(-1)
        if attention_mask is not None:
            token_logp = token_logp * attention_mask[:, 1:].to(token_logp.dtype)
        return token_logp

    def sequence_log_prob(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.token_log_probs(input_ids, attention_mask).sum(dim=-1)


class StudentPolicyWrapper(BasePolicyWrapper):
    """Trainable student initialized from the base generator; LoRA/PEFT by default."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str | None = None,
        *,
        use_lora: bool = True,
        lora_config: dict[str, Any] | None = None,
        adapter_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name_or_path, tokenizer_name_or_path, frozen=False, **kwargs)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True)
        elif use_lora:
            cfg = lora_config or {}
            target_modules = cfg.get("target_modules")
            lora_cfg = LoraConfig(
                r=int(cfg.get("r", 8)),
                lora_alpha=int(cfg.get("alpha", cfg.get("lora_alpha", 16))),
                lora_dropout=float(cfg.get("dropout", cfg.get("lora_dropout", 0.05))),
                bias=cfg.get("bias", "none"),
                target_modules=target_modules,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
        self.model.train()

    def save_pretrained(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class TeacherPolicyWrapper(BasePolicyWrapper):
    """Frozen base model plus one LoRA adapter used as an FDA or target teacher."""

    def __init__(self, base_model_name_or_path: str, adapter_path: str, tokenizer_name_or_path: str | None = None, *, teacher_name: str = "teacher", **kwargs: Any):
        super().__init__(base_model_name_or_path, tokenizer_name_or_path, frozen=False, **kwargs)
        self.teacher_name = teacher_name
        self.adapter_path = adapter_path
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        _freeze(self.model)


__all__ = ["BasePolicyWrapper", "StudentPolicyWrapper", "TeacherPolicyWrapper", "assert_tokenizer_compatible"]
