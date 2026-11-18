# -*- coding: utf-8 -*-
"""自回归文本生成。"""
from __future__ import annotations

import torch

from .config import DEVICE
from .data import CharCorpus


@torch.no_grad()
def generate_chars(model, corpus: CharCorpus, prompt: str, max_new_tokens: int = 100,
                   temperature: float = 1.0, top_k: int | None = None,
                   device: str = DEVICE) -> str:
    """编码 prompt 前向一次拿到状态，之后逐字符增量前向。"""
    model.eval()
    x = torch.tensor([corpus.encode(prompt)], device=device)   # (1, T)
    logits, state = model(x)
    logits = logits[:, -1, :]
    out = []
    for _ in range(max_new_tokens):
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        if top_k is not None:  # 只在概率最高的 k 个字符中采样
            k = min(top_k, probs.size(-1))
            thresh = torch.topk(probs, k).values[..., -1:]
            probs = torch.where(probs < thresh, torch.zeros_like(probs), probs)
            probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1)                        # (1, 1)
        out.append(nxt.item())
        logits, state = model(nxt, state)                        # 单步增量前向
        logits = logits[:, -1, :]
    return prompt + corpus.decode(out)
