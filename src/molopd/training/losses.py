from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CausalLMLossConfig:
    label_pad_token_id: int = -100


class CausalLMSFTLoss:
    """Explicit shifted-token negative log-likelihood for teacher LoRA SFT."""

    def __init__(self, cfg: CausalLMLossConfig | None = None) -> None:
        self.cfg = cfg or CausalLMLossConfig()

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid = (shift_labels != self.cfg.label_pad_token_id).float()
        if shift_labels.numel() == 0 or valid.sum().item() == 0:
            loss = logits.sum() * 0.0
            token_acc = logits.sum().detach() * 0.0
            ppl = torch.ones((), device=logits.device, dtype=logits.dtype)
            return {"loss": loss, "nll": loss.detach(), "perplexity": ppl, "token_acc": token_acc}
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.cfg.label_pad_token_id,
        )
        with torch.no_grad():
            token_acc = (shift_logits.argmax(dim=-1).eq(shift_labels) * valid.bool()).float().sum() / valid.sum().clamp_min(1.0)
            ppl = torch.exp(loss.detach().clamp(max=20.0))
        return {"loss": loss, "nll": loss, "perplexity": ppl, "token_acc": token_acc}
