# -*- coding: utf-8 -*-
"""用 Transformer 实现语言模型（GPT 风格 decoder-only），配合自研 BPE 分词扩大词表。

相比上一个字符级 LSTM 版本：
    - 语料：内置 1274 字符  ->  Shakespeare 1.1MB（自动下载，可 --data 换任意文本）
    - 词表：478 个字符      ->  16384 个 BPE 子词（从零训练的 byte-level BPE）
    - 主干：2 层 LSTM       ->  Transformer（因果自注意力 + Pre-LN + GELU MLP + 权重共享）
    - 规模：0.62M 参数      ->  约 57M 参数（12 层 / 12 头 / d_model=576 / 上下文 256）

用法：
    python transformer_lm.py                     # 默认配置训练 + 生成
    python transformer_lm.py --data my.txt       # 换用自己的语料（中文也可以）
    python transformer_lm.py --steps 6000        # 更多训练步数（默认 2000）
    python transformer_lm.py --retrain-bpe       # 重新训练分词器
"""
import argparse
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_URL = "https://fastly.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"
MERGES_PATH = os.path.join(DATA_DIR, "bpe_merges.json")

# 预分词：英文单词/数字/缩写，中文按单字切分，空白单独成片（GPT-2 风格）
PRETOKEN_RE = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]+"
    r"|[\u4e00-\u9fa5]"
    r"| ?[^\sA-Za-z0-9\u4e00-\u9fa5]+"
    r"|\s+"
)

FALLBACK_TEXT = (
    "\n".join([
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    ]) * 40
)


# ============================ 语料 ============================
def get_corpus(data_path=None):
    """优先用本地 --data，其次下载缓存，最后回落到内置小语料。"""
    if data_path:
        with open(data_path, encoding="utf-8") as f:
            return f.read()

    os.makedirs(DATA_DIR, exist_ok=True)
    local = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if os.path.exists(local) and os.path.getsize(local) > 100_000:
        with open(local, encoding="utf-8") as f:
            return f.read()

    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8")
            with open(local, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print(f"已下载语料 {len(text):,} 字符 -> {local}")
            return text
        except Exception as e:  # 网络抖动，重试
            print(f"下载失败({attempt}/5): {type(e).__name__} {str(e)[:50]}")
            time.sleep(2 * attempt)
    print("下载不可用，使用内置小语料（建议通过 --data 指定本地文本）")
    return FALLBACK_TEXT


# ============================ 从零实现 BPE ============================
class BPETokenizer:
    """byte-level BPE：256 个基础字节 + 逐轮合并出的新 token。"""

    def __init__(self, merges):
        self.merges = list(merges)
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:  # 按合并顺序重建每个 token 的字节串
            vocab.append(vocab[a] + vocab[b])
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size=4096, verbose=True):
        words = Counter()
        for piece in PRETOKEN_RE.findall(text):
            words[tuple(piece.encode("utf-8"))] += 1
        tokens = {w: list(w) for w in words}
        merges = []
        t0 = time.time()
        for step in range(max(0, vocab_size - 256)):
            stats = Counter()
            for w, freq in words.items():
                t = tokens[w]
                if len(t) < 2:
                    continue
                for pair in zip(t, t[1:]):
                    stats[pair] += freq
            if not stats:
                break
            best = max(stats.items(), key=lambda kv: (kv[1], kv[0]))[0]
            new_id = 256 + len(merges)
            for w in words:
                t = tokens[w]
                if len(t) < 2:
                    continue
                out, i = [], 0
                while i < len(t):
                    if i < len(t) - 1 and t[i] == best[0] and t[i + 1] == best[1]:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(t[i])
                        i += 1
                tokens[w] = out
            merges.append(best)
            if verbose and (step + 1) % 512 == 0:
                print(f"  BPE 合并 {step + 1}/{vocab_size - 256}  "
                      f"耗时 {time.time() - t0:.1f}s")
        if verbose:
            print(f"  BPE 完成：{len(merges)} 次合并，词表 {256 + len(merges)}，"
                  f"共 {time.time() - t0:.1f}s")
        return cls(merges)

    def encode_piece(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            ids[best_i:best_i + 2] = [256 + best_rank]
        self._cache[piece] = ids
        return ids

    def encode(self, text):
        out = []
        for piece in PRETOKEN_RE.findall(text):
            out.extend(self.encode_piece(piece))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls([tuple(m) for m in json.load(f)["merges"]])


# ============================ Transformer ============================
class CausalSelfAttention(nn.Module):
    """多头因果自注意力，用 SDPA 的 is_causal 一次完成掩码。"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN 残差结构：x + Attn(LN(x))，再 x + MLP(LN(x))。"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """decoder-only Transformer 语言模型。"""

    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4,
                 n_embd=192, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # 权重共享，省参数
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过 block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k).values[..., -1:]
                logits = torch.where(logits < thresh,
                                     torch.full_like(logits, float("-inf")), logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ============================ 训练 / 生成 ============================
def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size=32, iters=20):
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


def train(model, train_data, val_data, steps, block_size, batch_size,
          lr=3e-4, warmup=100, eval_every=250):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(1, steps + 1):
        # 线性 warmup + 余弦退火
        if step <= warmup:
            cur_lr = lr * step / warmup
        else:
            progress = (step - warmup) / max(1, steps - warmup)
            cur_lr = 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x, y = get_batch(train_data, block_size, batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0 or step == 1:
            val = estimate_loss(model, val_data, block_size)
            print(f"step {step:5d}  train {loss.item():.4f}  val {val:.4f}  "
                  f"ppl {math.exp(val):7.1f}  lr {cur_lr:.2e}  "
                  f"{time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="本地语料路径(UTF-8)")
    ap.add_argument("--vocab-size", type=int, default=16384,
                    help="BPE 词表大小（256 基础字节 + 合并数）")
    ap.add_argument("--block-size", type=int, default=256, help="上下文长度")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=576)
    ap.add_argument("--retrain-bpe", action="store_true", help="忽略缓存重建分词器")
    ap.add_argument("--bpe-chars", type=int, default=0,
                    help="训练 BPE 时采样多少字符，0 表示用全部语料")
    args = ap.parse_args()

    text = get_corpus(args.data)
    print(f"设备: {DEVICE}  语料 {len(text):,} 字符  torch {torch.__version__}")

    # 1) 分词器：有缓存直接加载，否则现场训练（约 1 分钟）
    os.makedirs(DATA_DIR, exist_ok=True)
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
    n_val = max(args.block_size + 1, len(data) // 20)
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
    train(model, train_data, val_data, args.steps, args.block_size, args.batch_size)

    # 5) 生成
    print("-" * 50 + "\n生成示例:")
    prompts = ["ROMEO:", "What is the", "The king "]
    for prompt in prompts:
        for temp in (0.8, 1.1):
            idx = torch.tensor([tok.encode(prompt)], device=DEVICE)
            out = model.generate(idx, 120, temperature=temp, top_k=40)
            print(f"\n[temp={temp}] {prompt}")
            print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
