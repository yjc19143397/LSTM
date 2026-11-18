# -*- coding: utf-8 -*-
"""自回归文本生成。"""
from __future__ import annotations

import torch

from .config import DEVICE
from .data import CharCorpus


def _sample_token(logits: torch.Tensor, temperature: float,
                  top_k: int | None) -> torch.Tensor:
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    if top_k is not None:  # 只在概率最高的 k 个字符中采样
        k = min(top_k, probs.size(-1))
        thresh = torch.topk(probs, k).values[..., -1:]
        probs = torch.where(probs < thresh, torch.zeros_like(probs), probs)
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1)


@torch.no_grad()
def generate_chars(model, corpus: CharCorpus, prompt: str, max_new_tokens: int = 100,
                   temperature: float = 1.0, top_k: int | None = None,
                   device: str = DEVICE, forward=None,
                   block_size: int | None = None) -> str:
    """生成字符。

    forward 为空时走有状态增量前向；提供时（如 Intel NPU）每次送入
    最近 block_size 个字符做固定形状前向。
    """
    model.eval()
    ids = corpus.encode(prompt)
    out = []

    if forward is None:
        x = torch.tensor([ids], device=device)                   # (1, T)
        logits, state = model(x)
        step_logits = logits[:, -1, :]
        for _ in range(max_new_tokens):
            nxt = _sample_token(step_logits, temperature, top_k)
            out.append(nxt.item())
            logits, state = model(nxt, state)                    # 单步增量前向
            step_logits = logits[:, -1, :]
    else:
        limit = block_size or len(ids) + max_new_tokens
        for _ in range(max_new_tokens):
            cond = torch.tensor([ids[-limit:]], device=device)
            logits = forward(cond)
            step_logits = logits[:, -1, :]
            nxt = _sample_token(step_logits, temperature, top_k)
            out.append(nxt.item())
            ids.append(nxt.item())

    return prompt + corpus.decode(out)
