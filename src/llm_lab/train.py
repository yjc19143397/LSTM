# -*- coding: utf-8 -*-
"""训练循环：字符级 LSTM 与 Transformer 语言模型。"""
from __future__ import annotations

import math
import time

import torch
from torch import nn

try:
    from tqdm import tqdm
except ImportError:  # 未安装 tqdm 时回退到按间隔打印
    tqdm = None

from .config import DEVICE
from .data import CharCorpus, get_batch


def _new_bar(total: int, desc: str):
    """有 tqdm 时返回进度条，否则返回 None，由调用方回退到日志打印。"""
    return tqdm(total=total, desc=desc, dynamic_ncols=True) if tqdm is not None else None


def train_char_lm(model: nn.Module, corpus: CharCorpus, steps: int, block_size: int = 48,
                  batch_size: int = 64, lr: float = 3e-3, device: str = DEVICE,
                  log_every: int = 250) -> float | None:
    """teacher forcing 训练：每个时间步都预测下一个字符。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = corpus.data
    loss_avg = None
    bar = _new_bar(steps, "char-lstm")
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
        if bar is not None:
            bar.update(1)
            bar.set_postfix(loss=f"{loss_avg:.4f}", ppl=f"{math.exp(loss_avg):.1f}")
        elif step == 1 or step % log_every == 0:
            print(f"step {step:4d}  loss {loss_avg:.4f}  ppl {math.exp(loss_avg):.1f}")
    if bar is not None:
        bar.close()
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
    bar = _new_bar(steps, "gpt")
    last_val = None
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
            last_val = estimate_loss(model, val_data, block_size, device=device)
            if bar is None:
                print(f"step {step:5d}  train {loss.item():.4f}  val {last_val:.4f}  "
                      f"ppl {math.exp(last_val):7.1f}  lr {cur_lr:.2e}  "
                      f"{time.time() - t0:.0f}s")

        if bar is not None:
            bar.update(1)
            post = {"loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.1e}"}
            if last_val is not None:
                post["val"] = f"{last_val:.4f}"
                post["ppl"] = f"{math.exp(last_val):.1f}"
            bar.set_postfix(post)
    if bar is not None:
        bar.close()
