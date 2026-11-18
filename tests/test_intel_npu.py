# -*- coding: utf-8 -*-
"""Intel NPU（OpenVINO）推理与 forward 接口测试。"""
import pytest
import torch

from llm_lab.intel_npu import OpenVINONPU, npu_available
from llm_lab.models import GPT


def test_npu_available_is_bool():
    assert isinstance(npu_available(), bool)


@pytest.mark.skipif(not npu_available(), reason="本机无 Intel NPU")
def test_npu_forward_matches_torch():
    model = GPT(32, 16, 1, 2, 16).eval()
    x = torch.randint(0, 32, (1, 10))
    npu = OpenVINONPU(model, 16)

    with torch.no_grad():
        ref = model(x)[0]
    got = npu(x)

    assert got.shape == ref.shape
    # NPU 以 fp16 执行，量化误差在 1e-3 量级
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_gpt_generate_accepts_forward():
    model = GPT(32, 16, 1, 2, 16).eval()
    idx = torch.zeros(1, 2, dtype=torch.long)
    out = model.generate(idx, 3, forward=lambda t: model(t)[0])
    assert out.shape == (1, 5)


def test_char_generate_accepts_forward():
    from llm_lab.data import CharCorpus
    from llm_lab.generate import generate_chars
    from llm_lab.models import CharLSTM

    corpus = CharCorpus("abc")
    model = CharLSTM(corpus.vocab_size, d_model=8, hidden=12, num_layers=1)
    out = generate_chars(model, corpus, "a", max_new_tokens=3,
                         device="cpu", forward=lambda t: model(t)[0],
                         block_size=8)
    assert out.startswith("a")
    assert len(out) == 4
