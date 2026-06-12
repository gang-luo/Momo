from __future__ import annotations

from pathlib import Path

import pytest
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@pytest.fixture()
def tiny_model_dir(tmp_path: Path) -> Path:
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "C": 4, "CCO": 5, "N": 6, "O": 7}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", bos_token="<bos>", eos_token="<eos>")
    cfg = GPT2Config(vocab_size=len(vocab), n_positions=32, n_embd=16, n_layer=1, n_head=2, bos_token_id=2, eos_token_id=3, pad_token_id=0)
    model = GPT2LMHeadModel(cfg)
    tokenizer.save_pretrained(tmp_path)
    model.save_pretrained(tmp_path)
    return tmp_path


@pytest.fixture()
def tiny_adapter_dir(tmp_path: Path, tiny_model_dir: Path) -> Path:
    model = GPT2LMHeadModel.from_pretrained(tiny_model_dir)
    peft_model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    adapter = tmp_path / "adapter"
    peft_model.save_pretrained(adapter)
    return adapter
