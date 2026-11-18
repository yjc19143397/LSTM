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


def _npu_available() -> bool:
    """导入 torch_npu 后 torch.npu 才注册；未安装或驱动不可用时返回 False。"""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available())


def resolve_device(name: str = "auto") -> str:
    """显式指定优先；auto 时按 NPU > CUDA > CPU 选择。

    这里的 ``npu`` 指昇腾（torch_npu）；Intel NPU 走 OpenVINO 推理，
    由 ``--intel-npu`` 控制，不进入该设备字符串。
    """
    if name == "npu":
        if not _npu_available():
            raise SystemExit(
                "未检测到昇腾 NPU（torch_npu）；Intel NPU 请使用 --intel-npu 做推理加速")
        return name
    if name and name != "auto":
        return name
    if _npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = resolve_device()


def disable_cudnn() -> None:
    """规避 PyTorch 2.11 + Python 3.14 + Windows 下 nn.LSTM 的退出崩溃。

    cuDNN 的 RNN 实现会在进程退出阶段触发 0xC0000409（环境问题，与模型逻辑无关）。
    禁用后走 PyTorch 原生 CUDA 实现即可根治，对纯 LSTM 训练的性能影响很小。
    """
    torch.backends.cudnn.enabled = False
