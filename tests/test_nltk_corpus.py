# -*- coding: utf-8 -*-
"""NLTK 英文词表的加载与接入测试。"""
from llm_lab.data import get_nltk_words


def test_nltk_words_loaded():
    words = get_nltk_words().splitlines()
    assert len(words) == 235_886
    assert words[0] == "A"
    assert "the" in words
    assert "computer" in words


def test_nltk_words_unique_and_nonempty():
    words = get_nltk_words().splitlines()
    assert len(words) == len(set(words))
    assert all(w == w.strip() and w for w in words)


def test_load_corpus_selects_nltk_words():
    from llm_lab.cli.main import _load_corpus

    assert _load_corpus(None, "nltk-words") == get_nltk_words()
