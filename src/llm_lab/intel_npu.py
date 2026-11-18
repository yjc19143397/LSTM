# -*- coding: utf-8 -*-
"""Intel NPU 推理支持：OpenVINO 固定形状前向，仅用于生成阶段。"""
from __future__ import annotations

import copy

import torch

try:
    import openvino as ov
except ImportError:  # 未安装时 npu_available() 返回 False
    ov = None


def npu_available() -> bool:
    """本机是否存在 OpenVINO 可用的 NPU（如 Intel AI Boost）。"""
    if ov is None:
        return False
    try:
        return "NPU" in ov.Core().available_devices
    except Exception:
        return False


class _LogitsOnly(torch.nn.Module):
    """把 (logits, loss/state) 统一成只返回 logits。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx):
        return self.model(idx)[0]


class OpenVINONPU:
    """把 (1, max_len) 的固定形状前向编译到 Intel NPU。

    OpenVINO 不支持训练，因此仅在生成阶段替换前向。输入右侧 pad 到固定长度，
    causal 注意力下真实位置的输出不受影响，返回时裁剪回真实长度。
    """

    def __init__(self, model: torch.nn.Module, max_len: int,
                 device: str = "NPU") -> None:
        if ov is None:
            raise RuntimeError("未安装 openvino，无法使用 Intel NPU")
        self.max_len = max_len
        cpu_model = copy.deepcopy(model).cpu().eval()
        example = torch.zeros(1, max_len, dtype=torch.long)
        ov_model = ov.convert_model(
            _LogitsOnly(cpu_model).eval(),
            example_input=example,
            input=[ov.PartialShape([1, max_len])],
        )
        if ov_model.is_dynamic():
            ov_model.reshape({0: [1, max_len]})
        self._compiled = ov.compile_model(ov_model, device)

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        length = idx.shape[1]
        if length > self.max_len:
            raise ValueError(f"输入长度 {length} 超过 NPU 固定形状 {self.max_len}")
        padded = torch.nn.functional.pad(idx.cpu(), (0, self.max_len - length))
        logits = self._compiled([padded.numpy()])[0]
        return torch.from_numpy(logits[:, :length]).to(idx.device)
