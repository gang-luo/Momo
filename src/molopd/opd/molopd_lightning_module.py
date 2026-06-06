from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from molopd.evaluators import RdkitEvaluator
from molopd.opd.losses import multi_teacher_opd_loss
from molopd.policies import BasePolicyWrapper, StudentPolicyWrapper, TeacherPolicyWrapper, assert_tokenizer_compatible
from molopd.sampling import OnPolicySampler
from molopd.training.replay_buffer import CsvReplayBuffer


class _InfiniteDummyDataset(Dataset):
    def __len__(self) -> int:  # Lightning uses max_steps to stop.
        return 10**9

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"idx": torch.tensor(idx)}


class MolOPDLightningModule(pl.LightningModule):
    """On-policy token-level multi-teacher MolOPD trainer.

    The only trainable parameters belong to ``self.student``. Base retention,
    FDA teacher and target teachers are frozen and queried on student samples.
    """

    def __init__(self, cfg: dict[str, Any] | Any):
        super().__init__()
        cfg = _to_plain(cfg)
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        model_cfg = cfg.get("model", cfg)
        teacher_cfg = cfg.get("teachers", {})
        lora_cfg = model_cfg.get("student_lora", {})
        tokenizer_path = model_cfg.get("tokenizer_name_or_path") or model_cfg["student_model_name_or_path"]
        common_kwargs = {
            "tokenizer_name_or_path": tokenizer_path,
            "torch_dtype": model_cfg.get("torch_dtype"),
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        }
        self.student = StudentPolicyWrapper(
            model_cfg["student_model_name_or_path"],
            use_lora=bool(model_cfg.get("use_lora_for_student", True)),
            lora_config=lora_cfg,
            adapter_path=model_cfg.get("student_adapter_path"),
            **common_kwargs,
        )
        self.base = BasePolicyWrapper(
            model_cfg.get("base_retention_model_name_or_path", model_cfg["student_model_name_or_path"]),
            frozen=True,
            **common_kwargs,
        )
        assert_tokenizer_compatible(self.student.tokenizer, self.base.tokenizer, "base_retention_model")

        fda_adapter = teacher_cfg.get("fda_adapter_path") or teacher_cfg.get("fda_teacher_adapter_path")
        self.fda_teacher = None
        if fda_adapter:
            self.fda_teacher = TeacherPolicyWrapper(
                model_cfg.get("base_teacher_model_name_or_path", model_cfg["student_model_name_or_path"]),
                fda_adapter,
                teacher_name="fda_teacher",
                **common_kwargs,
            )
            assert_tokenizer_compatible(self.student.tokenizer, self.fda_teacher.tokenizer, "fda_teacher")

        self.target_teacher_names: list[str] = []
        self.target_teachers = torch.nn.ModuleList()
        for i, item in enumerate(teacher_cfg.get("target_adapters", teacher_cfg.get("target_teacher_adapter_paths", []))):
            if isinstance(item, str):
                name, adapter = f"target_{i+1}", item
            else:
                name, adapter = item.get("name", f"target_{i+1}"), item.get("adapter_path")
            if adapter:
                teacher = TeacherPolicyWrapper(
                    model_cfg.get("base_teacher_model_name_or_path", model_cfg["student_model_name_or_path"]),
                    adapter,
                    teacher_name=name,
                    **common_kwargs,
                )
                assert_tokenizer_compatible(self.student.tokenizer, teacher.tokenizer, name)
                self.target_teachers.append(teacher)
                self.target_teacher_names.append(name)

        self.sampler = OnPolicySampler(self.student, self.student.tokenizer, prompt=cfg.get("sampling", {}).get("prompt", ""))
        out_dir = cfg.get("logging", {}).get("output_dir", "outputs/molopd")
        run_name = cfg.get("logging", {}).get("run_name", "run")
        self.replay = CsvReplayBuffer(f"{out_dir}/{run_name}")
        self.rdkit_eval = RdkitEvaluator() if cfg.get("evaluators", {}).get("use_rdkit_eval_for_logging", False) else None

    def train_dataloader(self):
        return DataLoader(_InfiniteDummyDataset(), batch_size=1, num_workers=0)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        sample_cfg = self.cfg.get("sampling", {})
        batch = self.sampler.sample(
            num_samples=int(sample_cfg.get("num_samples_per_step", 64)),
            max_new_tokens=int(sample_cfg.get("max_new_tokens", 128)),
            temperature=float(sample_cfg.get("temperature", 1.0)),
            top_p=float(sample_cfg.get("top_p", 0.95)),
            top_k=int(sample_cfg.get("top_k", 50)),
            keep_duplicates=bool(sample_cfg.get("keep_duplicates", False)),
            max_length=sample_cfg.get("max_length"),
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        student_logits = self.student.logits(input_ids, attention_mask)
        with torch.no_grad():
            base_logits = self.base.logits(input_ids, attention_mask)
            fda_logits = self.fda_teacher.logits(input_ids, attention_mask) if self.fda_teacher is not None else None
            target_logits = [teacher.logits(input_ids, attention_mask) for teacher in self.target_teachers]

        loss_cfg = self.cfg.get("loss", {})
        losses = multi_teacher_opd_loss(
            student_logits=student_logits,
            base_logits=base_logits,
            fda_teacher_logits=fda_logits,
            target_teacher_logits_list=target_logits,
            attention_mask=attention_mask,
            labels=labels,
            lambda_fda=float(loss_cfg.get("lambda_fda", 0.5)),
            lambda_base=float(loss_cfg.get("lambda_base", 0.1)),
            lambda_targets=loss_cfg.get("lambda_targets"),
            target_names=self.target_teacher_names,
            base_kl_direction=loss_cfg.get("base_kl_direction", "student_to_base"),
            lambda_nll=float(loss_cfg.get("lambda_nll", 0.0)),
            valid_ratio=float(batch["valid_ratio"]),
            unique_ratio=float(batch["unique_ratio"]),
        )
        metrics = {f"train/{k}": v for k, v in losses.items()}
        metrics["train/mean_length"] = attention_mask.sum(dim=1).float().mean()
        opt = self.optimizers(use_pl_optimizer=True)
        if opt is not None:
            metrics["train/learning_rate"] = torch.as_tensor(opt.param_groups[0]["lr"], device=self.device)
        self.log_dict(metrics, prog_bar=True, on_step=True, logger=True)
        self._log_samples(batch, input_ids, attention_mask)
        return losses["loss_total"]

    def _log_samples(self, batch: dict[str, Any], input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if int(self.global_step) % int(self.cfg.get("training", {}).get("sample_log_every_n_steps", 1)) != 0:
            return
        with torch.no_grad():
            logp_student = self.student.sequence_log_prob(input_ids, attention_mask).detach().cpu().tolist()
            logp_base = self.base.sequence_log_prob(input_ids, attention_mask).detach().cpu().tolist()
            logp_fda = self.fda_teacher.sequence_log_prob(input_ids, attention_mask).detach().cpu().tolist() if self.fda_teacher else [None] * len(batch["smiles"])
            target_logps = [t.sequence_log_prob(input_ids, attention_mask).detach().cpu().tolist() for t in self.target_teachers]
        rows = []
        for i, smiles in enumerate(batch["smiles"]):
            row = {
                "iteration": int(self.global_step),
                "raw_smiles": batch["raw_smiles"][i] if i < len(batch["raw_smiles"]) else smiles,
                "canonical_smiles": smiles,
                "valid": True,
                "logp_student": logp_student[i],
                "logp_base": logp_base[i],
                "logp_fda_teacher": logp_fda[i],
            }
            for j, vals in enumerate(target_logps):
                row[f"logp_{self.target_teacher_names[j]}"] = vals[i]
            rows.append(row)
        self.replay.write_iteration(int(self.global_step), rows)
        if self.rdkit_eval is not None:
            self.log_dict({f"rdkit/{k}": v for k, v in self.rdkit_eval(batch["smiles"]).items()}, on_step=True, logger=True)
        if self.logger is not None and hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.log({"sampled_smiles_examples": batch["smiles"][:8], "global_step": int(self.global_step)})
            except Exception:
                pass

    def configure_optimizers(self):
        train_cfg = self.cfg.get("training", {})
        opt = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=float(train_cfg.get("lr", 1e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        return opt

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Lightning stores weights in the checkpoint; also save a PEFT/HF-friendly adapter snapshot.
        save_dir = f"{self.cfg.get('logging', {}).get('output_dir', 'outputs/molopd')}/{self.cfg.get('logging', {}).get('run_name', 'run')}/student_step_{int(self.global_step)}"
        self.student.save_pretrained(save_dir)


def _to_plain(x: Any) -> Any:
    if hasattr(x, "items"):
        return {str(k): _to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    return x
