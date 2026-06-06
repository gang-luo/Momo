from __future__ import annotations

from transformers import AutoTokenizer

from molopd.policies import BasePolicyWrapper, TeacherPolicyWrapper


def test_base_and_teacher_load_forward(tiny_model_dir, tiny_adapter_dir):
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    batch = tok(["C"], return_tensors="pt")
    base = BasePolicyWrapper(str(tiny_model_dir))
    teacher = TeacherPolicyWrapper(str(tiny_model_dir), str(tiny_adapter_dir))
    assert base.logits(**batch).shape[-1] == len(tok)
    assert teacher.logits(**batch).shape[-1] == len(tok)
