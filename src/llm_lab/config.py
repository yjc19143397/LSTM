# -*- coding: utf-8 -*-
"""项目级常量：路径、设备与运行时开关。"""
from __future__ import annotations

import os

import torch

# src/llm_lab/config.py -> 上溯三级得到项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "tinyshakespeare.txt")
MERGES_PATH = os.path.join(DATA_DIR, "bpe_merges.json")
NLTK_WORDS_PATH = os.path.join(DATA_DIR, "nltk_words.txt")

DATA_URL = ("https://fastly.jsdelivr.net/gh/karpathy/char-rnn@master/"
            "data/tinyshakespeare/input.txt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def disable_cudnn() -> None:
    """规避 PyTorch 2.11 + Python 3.14 + Windows 下 nn.LSTM 的退出崩溃。

    cuDNN 的 RNN 实现会在进程退出阶段触发 0xC0000409（环境问题，与模型逻辑无关）。
    禁用后走 PyTorch 原生 CUDA 实现即可根治，对纯 LSTM 训练的性能影响很小。
    """
    torch.backends.cudnn.enabled = False
