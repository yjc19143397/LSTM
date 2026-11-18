# -*- coding: utf-8 -*-
"""训练循环：字符级 LSTM 与 Transformer 语言模型。"""
from __future__ import annotations

import math
import time

import torch
from torch import nn

from .config import DEVICE
from .data import CharCorpus, get_batch


def train_char_lm(model: nn.Module, corpus: CharCorpus, steps: int, block_size: int = 48,
                  batch_size: int = 64, lr: float = 3e-3, device: str = DEVICE,
                  log_every: int = 250) -> float | None:
    """teacher forcing 训练：每个时间步都预测下一个字符。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = corpus.data
    loss_avg = None
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)

        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # RNN 必备：防梯度爆炸
        opt.step()

        loss_avg = loss.item() if loss_avg is None else 0.9 * loss_avg + 0.1 * loss.item()
        if step == 1 or step % log_every == 0:
            print(f"step {step:4d}  loss {loss_avg:.4f}  ppl {math.exp(loss_avg):.1f}")
    return loss_avg


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, block_size: int,
                  batch_size: int = 32, iters: int = 20, device: str = DEVICE) -> float:
    """在固定批数上估计平均交叉熵损失。"""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


def train_transformer(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor,
                      steps: int, block_size: int, batch_size: int, lr: float = 3e-4,
                      warmup: int = 100, eval_every: int = 250,
                      device: str = DEVICE) -> None:
    """Transformer 训练：线性 warmup + 余弦退火，定期打印验证集困惑度。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(1, steps + 1):
        if step <= warmup:
            cur_lr = lr * step / warmup
        else:
            progress = (step - warmup) / max(1, steps - warmup)
            cur_lr = 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0 or step == 1:
            val = estimate_loss(model, val_data, block_size, device=device)
            print(f"step {step:5d}  train {loss.item():.4f}  val {val:.4f}  "
                  f"ppl {math.exp(val):7.1f}  lr {cur_lr:.2e}  "
                  f"{time.time() - t0:.0f}s")
