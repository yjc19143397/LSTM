# -*- coding: utf-8 -*-
"""Transformer 前向形状与生成测试。"""
import torch

from llm_lab.models import GPT


def test_gpt_forward_shapes_and_loss():
    model = GPT(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=16)
    idx = torch.randint(0, 32, (2, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(loss)


def test_gpt_generate_length():
    model = GPT(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16)
    idx = torch.randint(0, 32, (1, 4))
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 9)


def test_gpt_rejects_too_long_sequence():
    model = GPT(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
    try:
        model(torch.zeros(1, 9, dtype=torch.long))
    except AssertionError:
        return
    raise AssertionError("超过 block_size 的输入应当报错")
