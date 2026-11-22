# -*- coding: utf-8 -*-
"""项目级常量：路径、设备与运行时开关。"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# src/llm_lab/config.py -> 上溯三级得到项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "tinyshakespeare.txt")
MERGES_PATH = os.path.join(DATA_DIR, "bpe_merges.json")
NLTK_WORDS_PATH = os.path.join(DATA_DIR, "nltk_words.txt")

DATA_URL = ("https://fastly.jsdelivr.net/gh/karpathy/char-rnn@master/"
            "data/tinyshakespeare/input.txt")


@dataclass(frozen=True)
class _AccelBackend:
    """一个 PyTorch 加速后端：import 包名 -> torch.<attr> -> device 字符串。"""
    name: str             # --device 取值，也是传给 model.to() 的 device 字符串
    attr: str             # 探测 torch.<attr>；mps 特殊走 torch.backends.mps
    package: str | None   # 触发注册的 import 名；None 表示 PyTorch 内置无需安装
    label: str            # 人类可读的厂商/设备名
    pip: str              # 缺失时给出的安装提示


# auto 模式按此顺序探测，命中第一个可用的即返回；cuda / cpu 在最后兜底。
# 优先级把“专用 NPU / 国产算力卡”放在通用 CUDA 之前。
_BACKENDS = [
    _AccelBackend("npu",  "npu",  "torch_npu",  "昇腾 Ascend NPU",
                  "torch-npu"),
    _AccelBackend("mlu",  "mlu",  "torch_mlu",  "寒武纪 Cambricon MLU",
                  "torch-mlu"),
    _AccelBackend("musa", "musa", "torch_musa", "摩尔线程 Moore Threads MUSA",
                  "torch-musa"),
    _AccelBackend("dipu", "dipu", "torch_dipu",
                  "DIPU 统一算力层（海光/天数智芯/壁仞等）", "torch-dipu"),
    _AccelBackend("xpu",  "xpu",  "intel_extension_for_pytorch",
                  "Intel XPU GPU", "intel-extension-for-pytorch"),
    _AccelBackend("mps",  "mps",  None, "Apple Silicon MPS", "PyTorch 内置"),
]

# --device 可选值（auto / cpu / cuda 之外，再加上各加速后端）
DEVICE_CHOICES = ["auto", "cpu", "cuda"] + [b.name for b in _BACKENDS]


def _import_ok(package: str) -> bool:
    try:
        __import__(package)
    except ImportError:
        return False
    return True


def _backend_available(b: _AccelBackend) -> bool:
    """后端是否真正可用：先 import 触发注册，再查 torch.<attr>.is_available()。

    对 npu / mlu 等有 is_available() 的后端以其返回值为准；对 musa / dipu
    这类 import 成功即代表环境就绪的后端（参照 mmengine），若未注册
    is_available 则视为可用，避免漏掉真实硬件。
    """
    if b.package is not None and not _import_ok(b.package):
        return False
    if b.attr == "mps":
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    backend = getattr(torch, b.attr, None)
    if backend is None:
        return b.package is not None
    is_avail = getattr(backend, "is_available", None)
    return bool(is_avail is not None and is_avail())


def _backend_by_name(name: str) -> _AccelBackend | None:
    for b in _BACKENDS:
        if b.name == name:
            return b
    return None


def _npu_available() -> bool:
    """兼容旧调用：昇腾 NPU（torch_npu）是否可用。"""
    b = _backend_by_name("npu")
    return b is not None and _backend_available(b)


def _resolve_dipu_device() -> str:
    """DIPU 默认把设备 mock 成 cuda；未 mock 时取 torch_dipu.dipu.device.__dipu__。"""
    try:
        import torch_dipu  # noqa: F401
    except ImportError:
        return "cuda"
    if os.environ.get("DIPU_MOCK_CUDA") == "True":
        return "cuda"
    try:
        name = getattr(getattr(torch_dipu.dipu, "device", None), "__dipu__", None)
    except Exception:
        name = None
    if isinstance(name, str) and name:
        return name
    return "cuda"


def list_accelerators() -> list[tuple[str, str]]:
    """返回本机实际可用的加速后端 [(name, label), ...]，含 cuda。"""
    out = [(b.name, b.label) for b in _BACKENDS if _backend_available(b)]
    if torch.cuda.is_available():
        out.append(("cuda", "NVIDIA CUDA"))
    return out


def resolve_device(name: str = "auto") -> str:
    """解析运行设备字符串。

    auto 按 _BACKENDS 优先级 > cuda > cpu 选择；显式指定时校验可用性，
    不可用则给出清晰的安装提示并退出。

    Intel NPU（AI Boost）不是 PyTorch 张量设备，由 ``--intel-npu`` 通过
    OpenVINO 做推理加速，不进入这里的设备字符串。
    """
    if name == "auto" or not name:
        for b in _BACKENDS:
            if _backend_available(b):
                return _resolve_dipu_device() if b.name == "dipu" else b.name
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if name == "cpu":
        return "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("未检测到 CUDA 设备（torch.cuda.is_available() 为 False）")
        return "cuda"

    b = _backend_by_name(name)
    if b is None:
        raise SystemExit(f"未知设备 '{name}'；可选：{', '.join(DEVICE_CHOICES)}")
    if not _backend_available(b):
        hint = f"需安装 {b.pip}" if b.package is not None else ""
        extra = ""
        if name == "npu":
            extra = "；Intel NPU（AI Boost）请使用 --intel-npu 做推理加速"
        raise SystemExit(f"未检测到{b.label}（{hint}）{extra}")
    return _resolve_dipu_device() if name == "dipu" else name


DEVICE = resolve_device()


def disable_cudnn() -> None:
    """规避 PyTorch 2.11 + Python 3.14 + Windows 下 nn.LSTM 的退出崩溃。

    cuDNN 的 RNN 实现会在进程退出阶段触发 0xC0000409（环境问题，与模型逻辑无关）。
    禁用后走 PyTorch 原生 CUDA 实现即可根治，对纯 LSTM 训练的性能影响很小。
    """
    torch.backends.cudnn.enabled = False
