from __future__ import annotations

import torch

from molopd.opd.losses import multi_teacher_opd_loss


def test_kl_loss_scalar_masked_no_nan():
    b, t, v = 2, 5, 7
    student = torch.randn(b, t, v)
    base = torch.randn(b, t, v)
    fda = torch.randn(b, t, v)
    target = torch.randn(b, t, v)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    labels = torch.randint(0, v, (b, t))
    labels[mask == 0] = -100
    out = multi_teacher_opd_loss(
        student_logits=student,
        base_logits=base,
        fda_teacher_logits=fda,
        target_teacher_logits_list=[target],
        attention_mask=mask,
        labels=labels,
        lambda_targets={"target1": 1.0},
        target_names=["target1"],
    )
    assert out["loss_total"].ndim == 0
    assert torch.isfinite(out["loss_total"])
    assert torch.isfinite(out["student_entropy"])
