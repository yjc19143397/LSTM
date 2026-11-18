# -*- coding: utf-8 -*-
"""字符级 LSTM 语言模型：字符 -> Embedding -> 2 层 LSTM -> Linear -> 下一字符。"""
from __future__ import annotations

from torch import nn

from .lstm import MyLSTM


class CharLSTM(nn.Module):
    """字符级 LSTM 语言模型。

    scratch=True 时使用手写的 MyLSTM（输入 (T, B, D)），否则用 nn.LSTM
    的 CUDA 融合实现（训练快得多）。
    """

    def __init__(self, vocab_size: int, d_model: int = 64, hidden: int = 192,
                 num_layers: int = 2, scratch: bool = False):
        super().__init__()
        self.scratch = scratch
        self.emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(0.1)
        if scratch:
            self.rnn = MyLSTM(d_model, hidden, num_layers)
        else:
            # dropout 作用于层与层之间，抑制小语料上的过拟合
            self.rnn = nn.LSTM(d_model, hidden, num_layers, batch_first=True,
                               dropout=0.35 if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, state=None):
        x = self.drop(self.emb(idx))                 # (B, T, D)
        if self.scratch:
            out, state = self.rnn(x.permute(1, 0, 2), state)
            out = out.permute(1, 0, 2)
        else:
            out, state = self.rnn(x, state)          # (B, T, H)
        return self.head(out), state                 # logits (B, T, V)
