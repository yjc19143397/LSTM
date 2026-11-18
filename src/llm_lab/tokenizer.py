# -*- coding: utf-8 -*-
"""从零实现的 byte-level BPE 分词器。"""
from __future__ import annotations

import json
import time
from collections import Counter

from .data import PRETOKEN_RE


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
    def train(cls, text: str, vocab_size: int = 4096, verbose: bool = True) -> "BPETokenizer":
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

    def encode_piece(self, piece: str) -> list[int]:
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

    def encode(self, text: str) -> list[int]:
        out = []
        for piece in PRETOKEN_RE.findall(text):
            out.extend(self.encode_piece(piece))
        return out

    def decode(self, ids) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            return cls([tuple(m) for m in json.load(f)["merges"]])
