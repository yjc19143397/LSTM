# -*- coding: utf-8 -*-
"""用 LSTM 实现一个（小型）大语言模型：字符级语言模型 char-rnn。

原理与 GPT 完全一致——自回归地预测下一个 token，只是主干从 Transformer 换成 LSTM：
    字符 -> Embedding -> 2 层 LSTM -> Linear -> 下一字符的概率分布

完整流程：
    1. 语料分词（字符级）并构建词表
    2. 随机滑窗采样 (x, y)，y 是 x 右移一位（预测下一个字符）
    3. 训练：对所有时间步的预测计算交叉熵（teacher forcing）
    4. 生成：给定 prompt，携带 (h, c) 状态逐字符 temperature 采样，自回归续写

用法：
    python lstm_lm.py                    # 默认 nn.LSTM 训练 + 生成
    python lstm_lm.py --steps 3000       # 自定义训练步数
    python lstm_lm.py --scratch          # 换用 lstm.py 中手写的 MyLSTM（较慢，验证用）
"""
import argparse
import math

import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PyTorch 2.11 + Python 3.14 + Windows 下，nn.LSTM 的 cuDNN 实现会在进程退出阶段
# 触发 0xC0000409 崩溃（环境问题，与模型逻辑无关）。禁用 cuDNN 后走 PyTorch 原生
# CUDA 实现即可根治，对本脚本（纯 LSTM 训练）的性能影响很小。
torch.backends.cudnn.enabled = False

# ---------- 语料：内置经典古诗词（公版，约1300字符） ----------
CORPUS = "\n".join([
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


class CharCorpus:
    """字符级词表与整段语料的编码。"""

    def __init__(self, text: str):
        self.text = text
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


class CharLSTM(nn.Module):
    """字符级 LSTM 语言模型。"""

    def __init__(self, vocab_size, d_model=64, hidden=192, num_layers=2, scratch=False):
        super().__init__()
        self.scratch = scratch
        self.emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(0.1)
        if scratch:  # 复用 lstm.py 里手写的 MyLSTM（输入为 (T, B, D)）
            from lstm import MyLSTM
            self.rnn = MyLSTM(d_model, hidden, num_layers)
        else:        # nn.LSTM 有 CUDA 融合实现，训练快得多
            # dropout 作用于层与层之间，抑制小语料上的过拟合
            self.rnn = nn.LSTM(d_model, hidden, num_layers, batch_first=True,
                               dropout=0.35 if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, state=None):
        x = self.drop(self.emb(idx))                 # (B, T, D)
        if self.scratch:
            out, state = self.rnn(x.permute(1, 0, 2), state)
            out = out.permute(1, 0, 2)
        else:
            out, state = self.rnn(x, state)          # (B, T, H)
        return self.head(out), state                 # logits (B, T, V)


def train(model, corpus, steps, block_size=48, batch_size=64, lr=3e-3):
    """teacher forcing 训练：每个时间步都预测下一个字符。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = corpus.data
    loss_avg = None
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix]).to(DEVICE)
        y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(DEVICE)

        logits, _ = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # RNN 必备：防梯度爆炸
        opt.step()

        loss_avg = loss.item() if loss_avg is None else 0.9 * loss_avg + 0.1 * loss.item()
        if step == 1 or step % 250 == 0:
            print(f"step {step:4d}  loss {loss_avg:.4f}  ppl {math.exp(loss_avg):.1f}")


@torch.no_grad()
def generate(model, corpus, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    """自回归生成：编码 prompt 前向一次拿到状态，之后逐字符增量前向。"""
    model.eval()
    x = torch.tensor([corpus.encode(prompt)], device=DEVICE)   # (1, T)
    logits, state = model(x)
    logits = logits[:, -1, :]
    out = []
    for _ in range(max_new_tokens):
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        if top_k is not None:  # 只在概率最高的 k 个字符中采样
            k = min(top_k, probs.size(-1))
            thresh = torch.topk(probs, k).values[..., -1:]
            probs = torch.where(probs < thresh, torch.zeros_like(probs), probs)
            probs = probs / probs.sum()
        nxt = torch.multinomial(probs, 1)                        # (1, 1)
        out.append(nxt.item())
        logits, state = model(nxt, state)                        # 单步增量前向
        logits = logits[:, -1, :]
    return prompt + corpus.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000, help="训练步数")
    ap.add_argument("--scratch", action="store_true", help="使用手写 MyLSTM 代替 nn.LSTM")
    args = ap.parse_args()

    corpus = CharCorpus(CORPUS)
    model = CharLSTM(corpus.vocab_size, scratch=args.scratch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"设备: {DEVICE}  语料 {len(corpus.text)} 字符  词表 {corpus.vocab_size}  参数量 {n_params:,}")

    print("-" * 40 + "\n开始训练 ...")
    train(model, corpus, steps=args.steps)

    print("-" * 40 + "\n生成示例（不同温度的采样）:")
    prompts = ["床前明月光，", "春眠不觉晓，", "好雨知"]
    temps = [0.7, 1.0, 1.3]
    for prompt, temp in zip(prompts, temps):
        text = generate(model, corpus, prompt, max_new_tokens=100, temperature=temp, top_k=40)
        print(f"\n[temperature={temp}]  prompt: {prompt}")
        print(text)


if __name__ == "__main__":
    main()
