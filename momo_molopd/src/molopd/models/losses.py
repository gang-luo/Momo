from __future__ import annotations
import torch
import torch.nn.functional as F


def jsd_from_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.log_softmax(teacher_logits / temperature, dim=-1)
    sp = s.exp()
    tp = t.exp()
    m = 0.5 * (sp + tp)
    kl_sm = F.kl_div(s, m, reduction="none").sum(-1)
    kl_tm = F.kl_div(t, m, reduction="none").sum(-1)
    return 0.5 * (kl_sm + kl_tm)


def poe_teacher_logits(teacher_logits: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    w = torch.tensor(weights, device=teacher_logits[0].device, dtype=teacher_logits[0].dtype)
    w = w / w.sum().clamp(min=1e-8)
    stacked = torch.stack(teacher_logits, dim=0)
    return (stacked * w[:, None, None, None]).sum(dim=0)
