from __future__ import annotations
from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from molopd.models.losses import jsd_from_logits, poe_teacher_logits
from molopd.metrics.mol_metrics import compute_basic_generation_metrics


class MolTeacherLightningModule(pl.LightningModule):
    def __init__(self, model_name: str, lr: float = 2e-4, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05, t_max: int = 1000, eta_min: float = 1e-6):
        super().__init__()
        self.save_hyperparameters()
        base = AutoModelForCausalLM.from_pretrained(model_name)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg)

    def training_step(self, batch: dict[str, torch.Tensor], _: int):
        out = self.model(**batch)
        self.log("train/loss", out.loss, prog_bar=True)
        return out.loss

    def validation_step(self, batch: dict[str, torch.Tensor], _: int):
        out = self.model(**batch)
        self.log("val/loss", out.loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class MolOPDLightningModule(pl.LightningModule):
    def __init__(self, student_model_name: str, teacher_adapter_paths: list[str], teacher_weights: list[float], lr: float = 1e-4, t_max: int = 1000, eta_min: float = 1e-6, eval_num_samples: int = 128, eval_prompt: str = "C", eval_max_new_tokens: int = 64):
        super().__init__()
        self.save_hyperparameters()
        self.student = AutoModelForCausalLM.from_pretrained(student_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.teachers = []
        for path in teacher_adapter_paths:
            t_base = AutoModelForCausalLM.from_pretrained(student_model_name)
            t = PeftModel.from_pretrained(t_base, path)
            t.eval()
            for p in t.parameters():
                p.requires_grad = False
            self.teachers.append(t)

    def _opd_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        s_out = self.student(**batch)
        with torch.no_grad():
            teacher_logits = [t(**batch).logits for t in self.teachers]
            tgt_logits = poe_teacher_logits(teacher_logits, self.hparams.teacher_weights)
        token_jsd = jsd_from_logits(s_out.logits[:, :-1], tgt_logits[:, :-1])
        return token_jsd.mean()

    def training_step(self, batch: dict[str, torch.Tensor], _: int):
        loss = self._opd_loss(batch)
        self.log("train/opd_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _: int):
        loss = self._opd_loss(batch)
        self.log("val/opd_loss", loss, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], _: int):
        loss = self._opd_loss(batch)
        self.log("test/opd_loss", loss)

    def on_test_epoch_end(self):
        n = int(self.hparams.eval_num_samples)
        inputs = self.tokenizer(self.hparams.eval_prompt, return_tensors="pt").to(self.student.device)
        with torch.no_grad():
            outputs = self.student.generate(
                **inputs,
                max_new_tokens=int(self.hparams.eval_max_new_tokens),
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=n,
            )
        smiles = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        metrics = compute_basic_generation_metrics(smiles)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
