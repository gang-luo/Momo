from __future__ import annotations

from molopd.policies import StudentPolicyWrapper
from molopd.sampling import OnPolicySampler


def test_on_policy_sampler_fields_and_invalid_fallback(tiny_model_dir):
    student = StudentPolicyWrapper(str(tiny_model_dir), use_lora=False)
    sampler = OnPolicySampler(student, prompt="C")
    batch = sampler.sample(num_samples=2, max_new_tokens=2, keep_duplicates=False, max_length=16)
    for key in ["raw_smiles", "smiles", "valid", "input_ids", "attention_mask", "labels", "valid_ratio", "unique_ratio"]:
        assert key in batch
    assert batch["input_ids"].shape == batch["labels"].shape
