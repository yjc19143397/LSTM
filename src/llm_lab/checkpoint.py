# -*- coding: utf-8 -*-
"""预训练权重的保存与加载。"""
from __future__ import annotations

import os

import torch


def save_checkpoint(path: str, **payload) -> str:
    """把权重与元信息保存到 path，返回实际路径。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str) -> dict:
    """加载 checkpoint，统一映射到 CPU，避免设备不匹配。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到权重文件 {path}")
    return torch.load(path, map_location="cpu", weights_only=True)
