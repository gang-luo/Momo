from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def shift_mask(attention_mask: torch.Tensor | None, labels: torch.Tensor | None = None) -> torch.Tensor | None:
    if attention_mask is None and labels is None:
        return None
    if attention_mask is not None:
        mask = attention_mask[:, 1:].to(torch.float32)
    else:
        mask = torch.ones(labels[:, 1:].shape, device=labels.device, dtype=torch.float32)
    if labels is not None:
        mask = mask * (labels[:, 1:] != -100).to(mask.dtype)
    return mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(values.device, values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_token_kl(p_logits: torch.Tensor, q_logits: torch.Tensor, mask: torch.Tensor | None = None, *, temperature: float = 1.0) -> torch.Tensor:
    """KL(p || q) over token distributions, averaged over non-masked shifted positions."""
    p_log = F.log_softmax(p_logits[:, :-1, :] / temperature, dim=-1)
    q_log = F.log_softmax(q_logits[:, :-1, :] / temperature, dim=-1)
    p_prob = p_log.exp()
    kl = (p_prob * (p_log - q_log)).sum(dim=-1) * (temperature**2)
    return masked_mean(kl, mask)


def student_entropy(student_logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    logp = F.log_softmax(student_logits[:, :-1, :], dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    return masked_mean(entropy, mask)


def student_nll(student_logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    target = labels[:, 1:].clone()
    target[target == -100] = 0
    logp = F.log_softmax(student_logits[:, :-1, :], dim=-1)
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return masked_mean(nll, mask)


def multi_teacher_opd_loss(
    *,
    student_logits: torch.Tensor,
    base_logits: torch.Tensor,
    fda_teacher_logits: torch.Tensor | None,
    target_teacher_logits_list: list[torch.Tensor] | None,
    attention_mask: torch.Tensor | None,
    labels: torch.Tensor | None = None,
    lambda_fda: float = 0.5,
    lambda_base: float = 0.1,
    lambda_targets: dict[str, float] | list[float] | None = None,
    target_names: list[str] | None = None,
    base_kl_direction: str = "student_to_base",
    lambda_nll: float = 0.0,
    temperature: float = 1.0,
    valid_ratio: float = 1.0,
    unique_ratio: float = 1.0,
) -> dict[str, torch.Tensor]:
    mask = shift_mask(attention_mask, labels)
    zero = student_logits.sum() * 0.0
    losses: dict[str, torch.Tensor] = {}

    if fda_teacher_logits is not None and float(lambda_fda) != 0.0:
        losses["loss_fda"] = masked_token_kl(fda_teacher_logits, student_logits, mask, temperature=temperature)
    else:
        losses["loss_fda"] = zero

    if base_kl_direction == "student_to_base":
        losses["loss_base"] = masked_token_kl(student_logits, base_logits, mask, temperature=temperature)
    elif base_kl_direction == "base_to_student":
        losses["loss_base"] = masked_token_kl(base_logits, student_logits, mask, temperature=temperature)
    else:
        raise ValueError("base_kl_direction must be 'student_to_base' or 'base_to_student'")

    target_teacher_logits_list = target_teacher_logits_list or []
    target_names = target_names or [f"target_{i+1}" for i in range(len(target_teacher_logits_list))]
    if lambda_targets is None:
        target_weights = {name: 1.0 for name in target_names}
    elif isinstance(lambda_targets, dict):
        target_weights = {str(k): float(v) for k, v in lambda_targets.items()}
    else:
        target_weights = {name: float(w) for name, w in zip(target_names, lambda_targets)}

    target_total = zero
    for idx, logits in enumerate(target_teacher_logits_list):
        name = target_names[idx]
        key = f"loss_{name}" if name.startswith("target") else f"loss_target_{idx+1}"
        loss_i = masked_token_kl(logits, student_logits, mask, temperature=temperature)
        losses[key] = loss_i
        target_total = target_total + target_weights.get(name, 1.0) * loss_i

    nll = student_nll(student_logits, labels, mask) if labels is not None and float(lambda_nll) != 0.0 else zero
    losses["loss_nll"] = nll
    total = float(lambda_fda) * losses["loss_fda"] + float(lambda_base) * losses["loss_base"] + target_total + float(lambda_nll) * nll
    losses["loss_total"] = total
    losses["student_entropy"] = student_entropy(student_logits, mask)
    losses["valid_ratio"] = torch.as_tensor(float(valid_ratio), device=student_logits.device)
    losses["unique_ratio"] = torch.as_tensor(float(unique_ratio), device=student_logits.device)
    return losses
