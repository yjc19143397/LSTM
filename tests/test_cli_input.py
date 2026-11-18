# -*- coding: utf-8 -*-
"""命令行输入相关测试：--prompt、--interactive 与交互循环。"""
from llm_lab.cli.main import _interactive_char, _interactive_gpt, build_parser
from llm_lab.data import CharCorpus
from llm_lab.models import CharLSTM, GPT
from llm_lab.tokenizer import BPETokenizer


def test_parser_accepts_repeatable_prompts():
    args = build_parser().parse_args(
        ["char-lstm", "--prompt", "明月", "--prompt", "春风", "--interactive"])
    assert args.prompt == ["明月", "春风"]
    assert args.interactive is True


def test_transformer_parser_accepts_generation_options():
    args = build_parser().parse_args(
        ["transformer", "--prompt", "Once upon", "--max-new-tokens", "8",
         "--temperature", "0.5", "--top-k", "10"])
    assert args.prompt == ["Once upon"]
    assert args.max_new_tokens == 8
    assert args.temperature == 0.5
    assert args.top_k == 10


def test_interactive_char_exits_on_blank(monkeypatch, capsys):
    corpus = CharCorpus("abcabc")
    model = CharLSTM(corpus.vocab_size, d_model=8, hidden=12, num_layers=1)
    args = build_parser().parse_args(["char-lstm"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    _interactive_char(model, corpus, args)
    assert "退出" in capsys.readouterr().out


def test_interactive_char_reports_unknown_char(monkeypatch, capsys):
    corpus = CharCorpus("abc")
    model = CharLSTM(corpus.vocab_size, d_model=8, hidden=12, num_layers=1)
    args = build_parser().parse_args(["char-lstm"])
    inputs = iter(["z"])  # 第二次输入回落到空串，退出
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs, ""))
    _interactive_char(model, corpus, args)
    assert "词表外字符" in capsys.readouterr().out


def test_interactive_gpt_exits_on_quit(monkeypatch):
    tok = BPETokenizer.train("hello world", vocab_size=300, verbose=False)
    model = GPT(tok.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=8)
    args = build_parser().parse_args(["transformer"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": "quit")
    _interactive_gpt(model, tok, args)
