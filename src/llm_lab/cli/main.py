# -*- coding: utf-8 -*-
"""统一命令行入口：verify / char-lstm / transformer。"""
from __future__ import annotations

import argparse
import os

import torch

from ..checkpoint import load_checkpoint, save_checkpoint
from ..config import DEVICE, MERGES_PATH, ROOT_DIR, disable_cudnn, resolve_device
from ..data import POETRY_CORPUS, CharCorpus, get_corpus, get_nltk_words
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


def _load_corpus(data_path: str | None, corpus: str) -> str:
    """按 --data / --corpus 选择训练文本。"""
    if data_path:
        return get_corpus(data_path)
    if corpus == "nltk-words":
        return get_nltk_words()
    return get_corpus()


DEFAULT_CHAR_TEMPERATURES = [0.7, 1.0, 1.3]
DEFAULT_GPT_PROMPTS = ["ROMEO:", "What is the", "The king "]
DEFAULT_GPT_TEMPERATURES = [0.8, 1.1]


def _is_exit(text: str) -> bool:
    return not text.strip() or text.strip().lower() in {"exit", "quit"}


def _interactive_char(model, corpus: CharCorpus, args: argparse.Namespace,
                      device: str) -> None:
    """逐行读取 prompt，实时续写；空行或 exit/quit 退出。"""
    print("进入交互模式：输入 prompt 回车续写，空行 / exit 退出")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if _is_exit(prompt):
            return
        try:
            print(generate_chars(model, corpus, prompt,
                                 max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature,
                                 top_k=args.top_k, device=device))
        except KeyError as e:
            print(f"prompt 含有词表外字符 {e}，请换一段输入")


def _generate_gpt(model: GPT, tok: BPETokenizer, prompt: str,
                  args: argparse.Namespace, temperature: float, device: str) -> str:
    idx = torch.tensor([tok.encode(prompt)], device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=temperature,
                         top_k=args.top_k)
    return tok.decode(out[0].tolist())


def _interactive_gpt(model: GPT, tok: BPETokenizer, args: argparse.Namespace,
                     device: str) -> None:
    """逐行读取 prompt，实时续写；空行或 exit/quit 退出。"""
    print("进入交互模式：输入 prompt 回车续写，空行 / exit 退出")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if _is_exit(prompt):
            return
        print(_generate_gpt(model, tok, prompt, args, args.temperature, device))


def cmd_char_lstm(args: argparse.Namespace) -> None:
    """字符级 LSTM 语言模型：训练 + 生成。"""
    disable_cudnn()
    device = resolve_device(args.device)
    text = get_nltk_words() if args.corpus == "nltk-words" else POETRY_CORPUS
    corpus = CharCorpus(text)
    model = CharLSTM(corpus.vocab_size, scratch=args.scratch).to(device)

    if args.load:
        ckpt = load_checkpoint(args.load)
        if ckpt.get("scratch", args.scratch) != args.scratch:
            raise SystemExit("权重文件的 --scratch 设置与当前命令不一致")
        if ckpt.get("itos") != corpus.itos:
            raise SystemExit("权重文件的字符词表与当前语料不一致，请使用相同 --corpus")
        model.load_state_dict(ckpt["state_dict"])
        print(f"已加载预训练权重 {args.load}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备: {device}  语料 {len(corpus.text)} 字符  词表 {corpus.vocab_size}  "
          f"参数量 {n_params:,}")

    if args.steps > 0:
        print("-" * 40 + "\n开始训练 ...")
        train_char_lm(model, corpus, steps=args.steps, device=device)
        ckpt_path = save_checkpoint(
            args.checkpoint, model_type="char_lstm", state_dict=model.state_dict(),
            itos=corpus.itos, corpus=args.corpus, scratch=args.scratch)
        print(f"权重已保存 -> {ckpt_path}")

    if args.prompt:
        char_pairs = [(p, args.temperature) for p in args.prompt if p.strip()]
    else:
        # 用当前语料的开头作为默认 prompt，避免跨语料时出现词表外字符
        base = corpus.text[:12].strip() or corpus.text[:1]
        char_pairs = [(base, t) for t in DEFAULT_CHAR_TEMPERATURES]
    print("-" * 40 + "\n生成示例:")
    for prompt, temp in char_pairs:
        out_text = generate_chars(model, corpus, prompt,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=temp, top_k=args.top_k, device=device)
        print(f"\n[temperature={temp}]  prompt: {prompt}")
        print(out_text)

    if args.interactive:
        _interactive_char(model, corpus, args, device)


def cmd_transformer(args: argparse.Namespace) -> None:
    """BPE + Transformer 语言模型：训练 + 生成。"""
    device = resolve_device(args.device)
    text = _load_corpus(args.data, args.corpus)
    ckpt = load_checkpoint(args.load) if args.load else None

    # 1) 分词器：优先用权重自带的 merges，其次缓存，最后现场训练
    if ckpt is not None:
        tok = BPETokenizer(ckpt["merges"])
        model_cfg = ckpt["model_config"]
        print(f"从权重恢复分词器：词表 {tok.vocab_size}")
    else:
        os.makedirs(os.path.dirname(MERGES_PATH), exist_ok=True)
        if os.path.exists(MERGES_PATH) and not args.retrain_bpe:
            tok = BPETokenizer.load(MERGES_PATH)
            print(f"加载缓存分词器：词表 {tok.vocab_size}")
            if tok.vocab_size != args.vocab_size:
                print(f"注意：缓存词表 {tok.vocab_size} 与 --vocab-size "
                      f"{args.vocab_size} 不一致，以缓存为准")
            if args.bpe_extra != "none":
                print("提示：--bpe-extra 只在重建分词器时生效，需同时加 --retrain-bpe")
        else:
            print(f"训练 BPE 分词器（词表 {args.vocab_size}）...")
            bpe_text = text if args.bpe_chars <= 0 else text[:args.bpe_chars]
            if args.bpe_extra == "nltk-words":
                # 先喂英文词表，让高频单词优先被合并成整词 token
                bpe_text = get_nltk_words() + "\n" + bpe_text
                print("BPE 训练文本已加入 NLTK 英文词表")
            tok = BPETokenizer.train(bpe_text, args.vocab_size)
            tok.save(MERGES_PATH)
            print(f"分词器已缓存 -> {MERGES_PATH}")
        model_cfg = {"vocab_size": tok.vocab_size, "block_size": args.block_size,
                     "n_layer": args.n_layer, "n_head": args.n_head,
                     "n_embd": args.n_embd}

    block_size = model_cfg["block_size"]
    print(f"设备: {device}  语料 {len(text):,} 字符  torch {torch.__version__}")

    # 2) 编码语料并切分训练/验证集
    ids = tok.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    n_val = max(block_size + 2, len(data) // 20)
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"token 数 {len(data):,}（压缩比 {len(text) / len(data):.2f} 字符/token）"
          f"  训练 {len(train_data):,}  验证 {len(val_data):,}")

    # 3) 建模型
    model = GPT(model_cfg["vocab_size"], block_size, model_cfg["n_layer"],
                model_cfg["n_head"], model_cfg["n_embd"]).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        print(f"已加载预训练权重 {args.load}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_params:,}（{model_cfg['n_layer']} 层, "
          f"{model_cfg['n_head']} 头, d_model={model_cfg['n_embd']}）\n" + "-" * 50)

    # 4) 训练
    if args.steps > 0:
        train_transformer(model, train_data, val_data, args.steps,
                          block_size, args.batch_size, device=device)
        ckpt_path = save_checkpoint(args.checkpoint, model_type="gpt",
                                    state_dict=model.state_dict(),
                                    model_config=model_cfg, merges=tok.merges)
        print(f"权重已保存 -> {ckpt_path}")

    # 5) 生成
    if args.prompt:
        gpt_pairs = [(p, args.temperature) for p in args.prompt if p.strip()]
    else:
        gpt_pairs = [(p, t) for p in DEFAULT_GPT_PROMPTS
                     for t in DEFAULT_GPT_TEMPERATURES]
    print("-" * 50 + "\n生成示例:")
    for prompt, temp in gpt_pairs:
        print(f"\n[temp={temp}] {prompt}")
        print(_generate_gpt(model, tok, prompt, args, temp, device))

    if args.interactive:
        _interactive_gpt(model, tok, args, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-lab", description="从零实现的 LSTM / Transformer 语言模型")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify", help="验证手写 LSTM 与 nn.LSTM 的数值一致性")

    p_char = sub.add_parser("char-lstm", help="字符级 LSTM 语言模型：训练 + 生成")
    p_char.add_argument("--steps", type=int, default=2000, help="训练步数")
    p_char.add_argument("--scratch", action="store_true",
                        help="使用手写 MyLSTM 代替 nn.LSTM")
    p_char.add_argument("--corpus", choices=["poetry", "nltk-words"], default="poetry",
                        help="训练语料：内置古诗词或 NLTK 英文词表")
    p_char.add_argument("--prompt", action="append",
                        help="生成用的 prompt，可重复传入；不给则用内置示例")
    p_char.add_argument("--interactive", action="store_true",
                        help="训练后进入交互式续写")
    p_char.add_argument("--max-new-tokens", type=int, default=100,
                        help="每次生成的字符数")
    p_char.add_argument("--temperature", type=float, default=0.9,
                        help="--prompt / 交互模式使用的采样温度")
    p_char.add_argument("--top-k", type=int, default=40, help="top-k 采样")
    p_char.add_argument("--device", default="auto",
                        choices=["auto", "npu", "cuda", "cpu"],
                        help="运行设备，auto 按 NPU > CUDA > CPU 选择")
    p_char.add_argument("--checkpoint",
                        default=os.path.join(ROOT_DIR, "checkpoints", "char_lstm.pt"),
                        help="训练后保存权重的路径")
    p_char.add_argument("--load", default=None,
                        help="加载预训练权重（配合 --steps 0 可只生成）")

    p_tr = sub.add_parser("transformer", help="BPE + Transformer 语言模型：训练 + 生成")
    src = p_tr.add_mutually_exclusive_group()
    src.add_argument("--data", default=None, help="本地语料路径(UTF-8)")
    src.add_argument("--corpus", choices=["shakespeare", "nltk-words"],
                     default="shakespeare",
                     help="内置语料：TinyShakespeare 或 NLTK 英文词表")
    p_tr.add_argument("--bpe-extra", choices=["none", "nltk-words"], default="none",
                      help="重建 BPE 时额外加入 NLTK 词表，提升英文单词覆盖")
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
    p_tr.add_argument("--prompt", action="append",
                      help="生成用的 prompt，可重复传入；不给则用内置示例")
    p_tr.add_argument("--interactive", action="store_true",
                      help="训练后进入交互式续写")
    p_tr.add_argument("--max-new-tokens", type=int, default=120,
                      help="每次生成的 token 数")
    p_tr.add_argument("--temperature", type=float, default=0.9,
                      help="--prompt / 交互模式使用的采样温度")
    p_tr.add_argument("--top-k", type=int, default=40, help="top-k 采样")
    p_tr.add_argument("--device", default="auto",
                      choices=["auto", "npu", "cuda", "cpu"],
                      help="运行设备，auto 按 NPU > CUDA > CPU 选择")
    p_tr.add_argument("--checkpoint",
                      default=os.path.join(ROOT_DIR, "checkpoints", "transformer.pt"),
                      help="训练后保存权重的路径")
    p_tr.add_argument("--load", default=None,
                      help="加载预训练权重（配合 --steps 0 可只生成）")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {
        "verify": cmd_verify,
        "char-lstm": cmd_char_lstm,
        "transformer": cmd_transformer,
    }
    handlers[args.command](args)
