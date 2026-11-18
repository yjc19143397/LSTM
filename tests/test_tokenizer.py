# -*- coding: utf-8 -*-
"""BPE 分词器的训练、编解码与持久化测试。"""
from llm_lab.tokenizer import BPETokenizer


def test_bpe_roundtrip():
    text = "hello world, hello there. 你好，世界！"
    tok = BPETokenizer.train(text, vocab_size=300, verbose=False)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.vocab_size <= 300


def test_bpe_save_load(tmp_path):
    text = "the quick brown fox jumps over the lazy dog"
    tok = BPETokenizer.train(text, vocab_size=280, verbose=False)
    path = tmp_path / "merges.json"
    tok.save(str(path))

    loaded = BPETokenizer.load(str(path))
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
