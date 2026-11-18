# -*- coding: utf-8 -*-
"""统一命令行入口：verify / char-lstm / transformer。"""
from __future__ import annotations

import argparse
import os

import torch

from ..config import DEVICE, MERGES_PATH, disable_cudnn
from ..data import POETRY_CORPUS, CharCorpus, get_corpus
from ..demo import sine_forecast_demo
from ..generate import generate_chars
from ..models import CharLSTM, GPT, alignment_max_diff
from ..tokenizer import BPETokenizer
from ..train import train_char_lm, train_transformer


def cmd_verify(_args: argparse.Namespace) -> None:
    """验证手写 LSTM 与 nn.LSTM 的数值一致性，并跑正弦预测 demo。"""
    print(f"设备: {DEVICE}  torch: {torch.__version__}")
    for num_layers in (1, 2):
        diff = alignment_max_diff(num_layers)
        status = "PASS" if diff < 1e-6 else "FAIL"
        print(f"对齐验证 num_layers={num_layers}: {status} (max diff = {diff:.2e})")
    print("-" * 40)
    sine_forecast_demo()


def cmd_char_lstm(args: argparse.Namespace) -> None:
    """字符级 LSTM 语言模型：训练 + 生成。"""
    disable_cudnn()
    corpus = CharCorpus(POETRY_CORPUS)
    model = CharLSTM(corpus.vocab_size, scratch=args.scratch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备: {DEVICE}  语料 {len(corpus.text)} 字符  词表 {corpus.vocab_size}  "
          f"参数量 {n_params:,}")

    print("-" * 40 + "\n开始训练 ...")
    train_char_lm(model, corpus, steps=args.steps)

    print("-" * 40 + "\n生成示例（不同温度的采样）:")
    prompts = ["床前明月光，", "春眠不觉晓，", "好雨知"]
    temps = [0.7, 1.0, 1.3]
    for prompt, temp in zip(prompts, temps):
        text = generate_chars(model, corpus, prompt, max_new_tokens=100,
                              temperature=temp, top_k=40)
        print(f"\n[temperature={temp}]  prompt: {prompt}")
        print(text)


def cmd_transformer(args: argparse.Namespace) -> None:
    """BPE + Transformer 语言模型：训练 + 生成。"""
    text = get_corpus(args.data)
    print(f"设备: {DEVICE}  语料 {len(text):,} 字符  torch {torch.__version__}")

    # 1) 分词器：有缓存直接加载，否则现场训练（约 1 分钟）
    os.makedirs(os.path.dirname(MERGES_PATH), exist_ok=True)
    if os.path.exists(MERGES_PATH) and not args.retrain_bpe:
        tok = BPETokenizer.load(MERGES_PATH)
        print(f"加载缓存分词器：词表 {tok.vocab_size}")
        if tok.vocab_size != args.vocab_size:
            print(f"注意：缓存词表 {tok.vocab_size} 与 --vocab-size "
                  f"{args.vocab_size} 不一致，以缓存为准")
    else:
        print(f"训练 BPE 分词器（词表 {args.vocab_size}）...")
        bpe_text = text if args.bpe_chars <= 0 else text[:args.bpe_chars]
        tok = BPETokenizer.train(bpe_text, args.vocab_size)
        tok.save(MERGES_PATH)
        print(f"分词器已缓存 -> {MERGES_PATH}")

    # 2) 编码语料并切分训练/验证集
    ids = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    n_val = max(args.block_size + 2, len(data) // 20)
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"token 数 {len(data):,}（压缩比 {len(text) / len(data):.2f} 字符/token）"
          f"  训练 {len(train_data):,}  验证 {len(val_data):,}")

    # 3) 建模型
    model = GPT(tok.vocab_size, args.block_size, args.n_layer,
                args.n_head, args.n_embd).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_params:,}（{args.n_layer} 层, {args.n_head} 头, "
          f"d_model={args.n_embd}）\n" + "-" * 50)

    # 4) 训练
    train_transformer(model, train_data, val_data, args.steps,
                      args.block_size, args.batch_size)

    # 5) 生成
    print("-" * 50 + "\n生成示例:")
    prompts = ["ROMEO:", "What is the", "The king "]
    for prompt in prompts:
        for temp in (0.8, 1.1):
            idx = torch.tensor([tok.encode(prompt)], device=DEVICE)
            out = model.generate(idx, 120, temperature=temp, top_k=40)
            print(f"\n[temp={temp}] {prompt}")
            print(tok.decode(out[0].tolist()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-lab", description="从零实现的 LSTM / Transformer 语言模型")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify", help="验证手写 LSTM 与 nn.LSTM 的数值一致性")

    p_char = sub.add_parser("char-lstm", help="字符级 LSTM 语言模型：训练 + 生成")
    p_char.add_argument("--steps", type=int, default=2000, help="训练步数")
    p_char.add_argument("--scratch", action="store_true",
                        help="使用手写 MyLSTM 代替 nn.LSTM")

    p_tr = sub.add_parser("transformer", help="BPE + Transformer 语言模型：训练 + 生成")
    p_tr.add_argument("--data", default=None, help="本地语料路径(UTF-8)")
    p_tr.add_argument("--vocab-size", type=int, default=16384,
                      help="BPE 词表大小（256 基础字节 + 合并数）")
    p_tr.add_argument("--block-size", type=int, default=256, help="上下文长度")
    p_tr.add_argument("--batch-size", type=int, default=24)
    p_tr.add_argument("--steps", type=int, default=2000)
    p_tr.add_argument("--n-layer", type=int, default=12)
    p_tr.add_argument("--n-head", type=int, default=12)
    p_tr.add_argument("--n-embd", type=int, default=576)
    p_tr.add_argument("--retrain-bpe", action="store_true", help="忽略缓存重建分词器")
    p_tr.add_argument("--bpe-chars", type=int, default=0,
                      help="训练 BPE 时采样多少字符，0 表示用全部语料")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {
        "verify": cmd_verify,
        "char-lstm": cmd_char_lstm,
        "transformer": cmd_transformer,
    }
    handlers[args.command](args)
