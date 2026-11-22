# -*- coding: utf-8 -*-
"""多厂商加速设备解析测试：用 monkeypatch 模拟各后端包可用性（本机无真实硬件）。"""
import types

import pytest
import torch

from llm_lab import config
from llm_lab.config import (DEVICE_CHOICES, _BACKENDS, _backend_available,
                            _backend_by_name, list_accelerators, resolve_device)


def test_device_choices_cover_all_vendors():
    for name in ("auto", "cpu", "cuda", "npu", "mlu", "musa", "dipu", "xpu", "mps"):
        assert name in DEVICE_CHOICES


def test_backend_registry_completeness():
    assert {b.name for b in _BACKENDS} == {"npu", "mlu", "musa", "dipu", "xpu", "mps"}
    # 每个后端都有可读的厂商名与安装提示
    for b in _BACKENDS:
        assert b.label
        assert b.pip


def test_resolve_cpu_always_ok():
    assert resolve_device("cpu") == "cpu"


def test_resolve_cuda_depends_on_hardware():
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
    else:
        with pytest.raises(SystemExit):
            resolve_device("cuda")


def test_resolve_unknown_device_raises():
    with pytest.raises(SystemExit) as exc:
        resolve_device("tpu")
    assert "未知设备" in str(exc.value)


def test_unavailable_backend_raises_with_install_hint(monkeypatch):
    monkeypatch.setattr(config, "_import_ok", lambda pkg: False)
    with pytest.raises(SystemExit) as exc:
        resolve_device("mlu")
    msg = str(exc.value)
    assert "寒武纪" in msg
    assert "torch-mlu" in msg


def test_npu_unavailable_hints_intel_npu_flag(monkeypatch):
    monkeypatch.setattr(config, "_import_ok", lambda pkg: False)
    with pytest.raises(SystemExit) as exc:
        resolve_device("npu")
    assert "intel-npu" in str(exc.value)


@pytest.mark.parametrize("name", ["npu", "mlu", "musa", "xpu"])
def test_explicit_backend_resolves_when_available(monkeypatch, name):
    b = _backend_by_name(name)
    monkeypatch.setattr(config, "_import_ok", lambda pkg, _b=b: pkg == _b.package)
    monkeypatch.setattr(torch, b.attr,
                        types.SimpleNamespace(is_available=lambda: True),
                        raising=False)
    assert resolve_device(name) == name


@pytest.mark.parametrize("name", ["npu", "mlu", "musa", "xpu"])
def test_auto_picks_available_backend(monkeypatch, name):
    b = _backend_by_name(name)
    monkeypatch.setattr(config, "_import_ok", lambda pkg, _b=b: pkg == _b.package)
    monkeypatch.setattr(torch, b.attr,
                        types.SimpleNamespace(is_available=lambda: True),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == name


def test_auto_priority_higher_backend_wins(monkeypatch):
    # 同时模拟 npu 与 mlu 可用，auto 应选优先级更高的 npu
    monkeypatch.setattr(config, "_import_ok",
                        lambda pkg: pkg in {"torch_npu", "torch_mlu"})
    monkeypatch.setattr(torch, "npu",
                        types.SimpleNamespace(is_available=lambda: True),
                        raising=False)
    monkeypatch.setattr(torch, "mlu",
                        types.SimpleNamespace(is_available=lambda: True),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "npu"


def test_backend_unavailable_when_is_available_false(monkeypatch):
    monkeypatch.setattr(config, "_import_ok", lambda pkg: True)
    monkeypatch.setattr(torch, "mlu",
                        types.SimpleNamespace(is_available=lambda: False),
                        raising=False)
    assert _backend_available(_backend_by_name("mlu")) is False


def test_mps_available_via_torch_backends(monkeypatch):
    monkeypatch.setattr(config, "_import_ok", lambda pkg: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("mps") == "mps"
    assert resolve_device("auto") == "mps"


def test_dipu_routes_through_resolver(monkeypatch):
    b = _backend_by_name("dipu")
    monkeypatch.setattr(config, "_import_ok", lambda pkg, _b=b: pkg == _b.package)
    monkeypatch.setattr(config, "_resolve_dipu_device", lambda: "dipu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("dipu") == "dipu"
    assert resolve_device("auto") == "dipu"


def test_resolve_dipu_device_falls_back_without_torch_dipu():
    # 本机无 torch_dipu：应回退为 cuda（DIPU 默认 mock cuda）
    assert config._resolve_dipu_device() == "cuda"


def test_list_accelerators_includes_simulated(monkeypatch):
    monkeypatch.setattr(config, "_import_ok", lambda pkg: pkg == "torch_mlu")
    monkeypatch.setattr(torch, "mlu",
                        types.SimpleNamespace(is_available=lambda: True),
                        raising=False)
    names = dict(list_accelerators())
    assert "mlu" in names
