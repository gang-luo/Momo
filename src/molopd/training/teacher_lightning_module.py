from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from molopd.training.config import pl
from molopd.training.losses import CausalLMLossConfig, CausalLMSFTLoss


def _ensure_pad(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


class MolTeacherSFTLightningModule(pl.LightningModule):
    """Lightning module for FDA/target teacher construction by LoRA SFT.

    This module owns only training logic. Model paths, data paths, LoRA settings,
    optimizer/scheduler settings, trainer settings and logging are supplied by
    separated YAML sections and wired by the train entrypoint.
    """

    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        lora_cfg: dict[str, Any],
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
        loss_cfg: dict[str, Any] | None = None,
        teacher_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.teacher_cfg = teacher_cfg or {}
        base_model = model_cfg["base_model_name_or_path"]
        tokenizer_path = model_cfg.get("tokenizer_name_or_path") or base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=bool(model_cfg.get("trust_remote_code", False)))
        _ensure_pad(self.tokenizer)
        base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=bool(model_cfg.get("trust_remote_code", False)))
        self.model = get_peft_model(
            base,
            LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", lora_cfg.get("lora_alpha", 32))),
                lora_dropout=float(lora_cfg.get("dropout", lora_cfg.get("lora_dropout", 0.05))),
                target_modules=lora_cfg.get("target_modules"),
                bias=lora_cfg.get("bias", "none"),
                task_type="CAUSAL_LM",
            ),
        )
        self.loss_fn = CausalLMSFTLoss(CausalLMLossConfig(**(loss_cfg or {})))
        self._skip_optimizer_step_due_to_oom = False

    def forward(self, **batch: torch.Tensor):
        return self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))

    def _tensor_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        tensors = self._tensor_batch(batch)
        out = self.model(input_ids=tensors["input_ids"], attention_mask=tensors.get("attention_mask"))
        loss_dict = self.loss_fn(out.logits, tensors["labels"])
        for key, value in loss_dict.items():
            self.log(f"{stage}/{key}", value, prog_bar=(key == "loss"), on_step=(stage == "train"), on_epoch=True, logger=True)
        if stage in {"val", "test"} and "smiles" in batch:
            self.log(f"{stage}/num_smiles", torch.tensor(float(len(batch["smiles"])), device=self.device), on_step=False, on_epoch=True)
        loss = loss_dict["loss"]
        if not torch.isfinite(loss.detach()):
            self.log(f"{stage}/skip_nonfinite_loss", torch.tensor(1.0, device=self.device), on_step=(stage == "train"), on_epoch=True)
            return self._zero_loss() if stage == "train" else loss.detach()
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable LoRA parameters found for teacher SFT.")
        opt_name = str(self.optimizer_cfg.get("name", "adamw")).lower()
        if opt_name != "adamw":
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.optimizer_cfg.get("lr", 2e-4)),
            weight_decay=float(self.optimizer_cfg.get("weight_decay", 0.0)),
            betas=tuple(self.optimizer_cfg.get("betas", (0.9, 0.999))),
            eps=float(self.optimizer_cfg.get("eps", 1e-8)),
        )
        if not self.scheduler_cfg or str(self.scheduler_cfg.get("name", "none")).lower() in {"", "none"}:
            return optimizer
        name = str(self.scheduler_cfg.get("name", "cosine")).lower()
        if name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", 1000)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 1e-6)),
            )
        elif name == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(self.scheduler_cfg.get("milestones", [1000, 2000])),
                gamma=float(self.scheduler_cfg.get("gamma", 0.1)),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {name}")
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _zero_loss(self) -> torch.Tensor:
        return next(self.model.parameters()).sum() * 0.0

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as exc:
            if not self._is_oom_error(exc):
                raise
            self._skip_optimizer_step_due_to_oom = True
            self.log("train/skip_backward_oom", torch.tensor(1.0, device=self.device), on_step=True, on_epoch=True)
            for p in self.model.parameters():
                p.grad = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure) -> None:
        if self._skip_optimizer_step_due_to_oom:
            self._skip_optimizer_step_due_to_oom = False
            optimizer_closure()
            optimizer.zero_grad(set_to_none=True)
            self.log("train/skip_optim_step_oom", torch.tensor(1.0, device=self.device), on_step=True, on_epoch=True)
            return
        optimizer.step(closure=optimizer_closure)

    def save_adapter(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def on_train_end(self) -> None:
        output_dir = self.model_cfg.get("output_adapter_dir") or self.teacher_cfg.get("output_adapter_dir")
        if output_dir:
            self.save_adapter(output_dir)
