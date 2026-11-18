# -*- coding: utf-8 -*-
"""字符级语言模型的语料编码与前向形状测试。"""
import torch

from llm_lab.data import POETRY_CORPUS, CharCorpus, get_batch
from llm_lab.models import CharLSTM


def test_char_corpus_roundtrip():
    corpus = CharCorpus("abcabc")
    assert corpus.vocab_size == 3
    assert corpus.decode(corpus.encode("cab")) == "cab"


def test_char_lstm_forward_shapes():
    corpus = CharCorpus(POETRY_CORPUS)
    model = CharLSTM(corpus.vocab_size, d_model=16, hidden=24, num_layers=1)
    logits, _ = model(torch.zeros(2, 8, dtype=torch.long))
    assert logits.shape == (2, 8, corpus.vocab_size)


def test_char_lstm_scratch_forward_shapes():
    corpus = CharCorpus("abcabc")
    model = CharLSTM(corpus.vocab_size, d_model=8, hidden=12, num_layers=2,
                     scratch=True)
    logits, _ = model(torch.zeros(1, 5, dtype=torch.long))
    assert logits.shape == (1, 5, corpus.vocab_size)


def test_get_batch_shifted_targets():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=10, batch_size=3, device="cpu")
    assert x.shape == y.shape == (3, 10)
    assert torch.equal(x[:, 1:], y[:, :-1])
