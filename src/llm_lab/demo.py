# -*- coding: utf-8 -*-
"""训练 demo：用手写 LSTM 预测正弦序列的下一个点。"""
from __future__ import annotations

import torch
from torch import nn

from .models import MyLSTM


def sine_forecast_demo(epochs: int = 300, window: int = 20, hidden: int = 32,
                       lr: float = 0.01) -> float:
    """给定过去 window 个点的正弦窗口，预测下一个点，返回最终 MSE。"""
    torch.manual_seed(0)

    # 构造数据：滑窗切分正弦序列
    t = torch.arange(0, 200, dtype=torch.float32)
    wave = torch.sin(t * 0.1)
    xs, ys = [], []
    for i in range(len(wave) - window):
        xs.append(wave[i:i + window])
        ys.append(wave[i + window])
    X = torch.stack(xs).unsqueeze(-1)   # (N, window, 1)
    y = torch.stack(ys)                 # (N,)
    X = X.permute(1, 0, 2)              # (window, N, 1)

    lstm = MyLSTM(1, hidden)
    head = nn.Linear(hidden, 1)

    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        outputs, _ = lstm(X)            # (window, N, hidden)
        pred = head(outputs[-1]).squeeze(-1)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:3d}  loss = {loss.item():.6f}")

    with torch.no_grad():
        pred = head(lstm(X)[0][-1]).squeeze(-1)
        mse = loss_fn(pred, y).item()
        print(f"最终 MSE = {mse:.6f}")
        print("前5个样本 预测 vs 真实:")
        for p, gt in zip(pred[:5].tolist(), y[:5].tolist()):
            print(f"  {p:+.4f}  vs  {gt:+.4f}")
    return mse
