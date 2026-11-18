# -*- coding: utf-8 -*-
"""语料加载、字符级词表与语言模型批采样。"""
from __future__ import annotations

import os
import re
import time
import urllib.request

import torch

from .config import DATA_DIR, DATA_URL, NLTK_WORDS_PATH, SHAKESPEARE_PATH

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

# 字符级 LSTM 默认语料：内置经典古诗词（公版，约 1300 字符）
POETRY_CORPUS = "\n".join([
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
    "鹅鹅鹅，曲项向天歌。白毛浮绿水，红掌拨清波。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "山中相送罢，日暮掩柴扉。春草明年绿，王孙归不归。",
    "君自故乡来，应知故乡事。来日绮窗前，寒梅著花未。",
    "终南阴岭秀，积雪浮云端。林表明霁色，城中增暮寒。",
    "移舟泊烟渚，日暮客愁新。野旷天低树，江清月近人。",
    "松下问童子，言师采药去。只在此山中，云深不知处。",
    "危楼高百尺，手可摘星辰。不敢高声语，恐惊天上人。",
    "白发三千丈，缘愁似个长。不知明镜里，何处得秋霜。",
    "小时不识月，呼作白玉盘。又疑瑶台镜，飞在青云端。",
    "向晚意不适，驱车登古原。夕阳无限好，只是近黄昏。",
    "解落三秋叶，能开二月花。过江千尺浪，入竹万竿斜。",
    "小娃撑小艇，偷采白莲回。不解藏踪迹，浮萍一道开。",
    "绿蚁新醅酒，红泥小火炉。晚来天欲雪，能饮一杯无。",
    "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
    "人闲桂花落，夜静春山空。月出惊山鸟，时鸣春涧中。",
    "木末芙蓉花，山中发红萼。涧户寂无人，纷纷开且落。",
    "好雨知时节，当春乃发生。随风潜入夜，润物细无声。",
    "日暮苍山远，天寒白屋贫。柴门闻犬吠，风雪夜归人。",
    "苍苍竹林寺，杳杳钟声晚。荷笠带斜阳，青山独归远。",
    "泠泠七弦上，静听松风寒。古调虽自爱，今人多不弹。",
    "怀君属秋夜，散步咏凉天。空山松子落，幽人应未眠。",
    "三日入厨下，洗手作羹汤。未谙姑食性，先遣小姑尝。",
    "寥落古行宫，宫花寂寞红。白头宫女在，闲坐说玄宗。",
    "北斗七星高，哥舒夜带刀。至今窥牧马，不敢过临洮。",
    "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。",
    "远看山有色，近听水无声。春去花还在，人来鸟不惊。",
    "林暗草惊风，将军夜引弓。平明寻白羽，没在石棱中。",
    "月黑雁飞高，单于夜遁逃。欲将轻骑逐，大雪满弓刀。",
    "岭外音书绝，经冬复历春。近乡情更怯，不敢问来人。",
    "十年磨一剑，霜刃未曾试。今日把示君，谁有不平事。",
    "故国三千里，深宫二十年。一声何满子，双泪落君前。",
    "江上往来人，但爱鲈鱼美。君看一叶舟，出没风波里。",
    "墙角数枝梅，凌寒独自开。遥知不是雪，为有暗香来。",
    "众鸟高飞尽，孤云独去闲。相看两不厌，只有敬亭山。",
    "天下伤心处，劳劳送客亭。春风知别苦，不遣柳条青。",
    "君家何处住，妾住在横塘。停船暂借问，或恐是同乡。",
    "长江悲已滞，万里念将归。况属高风晚，山山黄叶飞。",
    "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。",
    "前不见古人，后不见来者。念天地之悠悠，独怆然而涕下。",
    "行到水穷处，坐看云起时。兴来每独往，胜事空自知。",
    "嫁得瞿塘贾，朝朝误妾期。早知潮有信，嫁与弄潮儿。",
    "打起黄莺儿，莫教枝上啼。啼时惊妾梦，不得到辽西。",
    "洛阳访才子，江岭作流人。闻说梅花早，何如此地春。",
])


def get_corpus(data_path: str | None = None) -> str:
    """优先用本地 --data，其次读取 data/ 缓存，最后回落到内置小语料。"""
    if data_path:
        with open(data_path, encoding="utf-8") as f:
            return f.read()

    if os.path.exists(SHAKESPEARE_PATH) and os.path.getsize(SHAKESPEARE_PATH) > 100_000:
        with open(SHAKESPEARE_PATH, encoding="utf-8") as f:
            return f.read()

    os.makedirs(DATA_DIR, exist_ok=True)
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8")
            with open(SHAKESPEARE_PATH, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print(f"已下载语料 {len(text):,} 字符 -> {SHAKESPEARE_PATH}")
            return text
        except Exception as e:  # 网络抖动，重试
            print(f"下载失败({attempt}/5): {type(e).__name__} {str(e)[:50]}")
            time.sleep(2 * attempt)
    print("下载不可用，使用内置小语料（建议通过 --data 指定本地文本）")
    return FALLBACK_TEXT


def get_nltk_words(path: str | None = None) -> str:
    """读取 NLTK words 词表（每行一个词），返回按换行拼接的文本。

    词表来自 nltk.corpus.words，共 235,886 个英文词，已导出到 data/nltk_words.txt。
    """
    path = path or NLTK_WORDS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 NLTK 词表 {path}，请参考 README 重新生成 data/nltk_words.txt")
    with open(path, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return "\n".join(words)


class CharCorpus:
    """字符级词表与整段语料的编码。"""

    def __init__(self, text: str):
        self.text = text
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """随机切出 batch_size 个长度为 block_size 的窗口，y 为 x 右移一位。"""
    if len(data) <= block_size + 1:
        raise ValueError(
            f"语料只有 {len(data)} 个 token，不足以切出 block_size={block_size} 的窗口")
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
