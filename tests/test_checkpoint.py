# -*- coding: utf-8 -*-
"""权重保存/加载与设备解析测试。"""
import pytest
import torch

from llm_lab.checkpoint import load_checkpoint, save_checkpoint
from llm_lab.config import DEVICE_CHOICES, resolve_device
from llm_lab.models import GPT


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "gpt.pt"
    model = GPT(32, 16, 1, 2, 16)
    save_checkpoint(
        str(path),
        model_type="gpt",
        state_dict=model.state_dict(),
        model_config={"vocab_size": 32, "block_size": 16, "n_layer": 1,
                      "n_head": 2, "n_embd": 16},
    )

    ckpt = load_checkpoint(str(path))
    assert ckpt["model_type"] == "gpt"
    assert ckpt["model_config"]["n_embd"] == 16
    for key, value in model.state_dict().items():
        assert torch.equal(value, ckpt["state_dict"][key])


def test_load_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("definitely_not_exists.pt")


def test_resolve_device_explicit():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_npu_requires_torch_npu():
    from llm_lab.config import _npu_available

    if _npu_available():
        assert resolve_device("npu") == "npu"
    else:
        with pytest.raises(SystemExit):
            resolve_device("npu")


def test_resolve_device_auto():
    assert resolve_device("auto") in (set(DEVICE_CHOICES) - {"auto"})
