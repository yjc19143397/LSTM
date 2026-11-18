# llm-lab

从零实现的小型语言模型实验项目：手写 LSTM 门控单元、字符级 char-rnn、byte-level BPE 分词器与 GPT 风格 Transformer。全部基于 PyTorch，代码按功能分层，便于阅读与扩展。

## 项目结构

```
lstm/
├── data/                     # 语料与分词器缓存
│   ├── tinyshakespeare.txt   # 1.1MB 英文语料
│   ├── nltk_words.txt        # NLTK 英文词表（235,886 词）
│   └── bpe_merges.json       # BPE 合并规则缓存
├── src/llm_lab/
│   ├── config.py             # 路径、设备、cuDNN 开关
│   ├── data.py               # 语料加载、字符词表、批采样
│   ├── tokenizer.py          # byte-level BPE 分词器
│   ├── demo.py               # 正弦序列预测 demo
│   ├── train.py              # 训练循环（tqdm 进度条）
│   ├── generate.py           # 字符级自回归生成
│   ├── checkpoint.py         # 预训练权重保存/加载
│   ├── models/
│   │   ├── lstm.py           # 手写 LSTM 单元与多层堆叠、对齐验证
│   │   ├── char_lm.py        # 字符级 LSTM 语言模型
│   │   └── transformer.py    # GPT：因果自注意力 / Pre-LN / 权重共享
│   └── cli/main.py           # 统一命令行入口
├── checkpoints/              # 训练产物（git 忽略）
├── tests/                    # pytest 单元测试
├── run.py                    # 免安装运行入口
├── pyproject.toml            # 包配置与 pytest 配置
└── requirements.txt
```

## 快速开始

无需安装，直接运行：

```powershell
python run.py verify                         # 手写 LSTM 与 nn.LSTM 数值对齐 + 正弦预测 demo
python run.py char-lstm --steps 2000         # 字符级 LSTM（内置古诗词语料）
python run.py char-lstm --scratch            # 用自己手写的 MyLSTM 训练（较慢，验证用）
python run.py transformer --steps 2000       # BPE + Transformer（TinyShakespeare）
```

用 NLTK 英文词库训练：

```powershell
python run.py char-lstm --corpus nltk-words                    # 词表作为字符级语料
python run.py transformer --corpus nltk-words --steps 2000     # 词表作为 Transformer 语料
python run.py transformer --retrain-bpe --bpe-extra nltk-words # 重建 BPE 时优先合并英文整词
```

自定义生成输入：

```powershell
python run.py char-lstm --prompt "床前明月光，" --max-new-tokens 60   # 自定义 prompt
python run.py char-lstm --interactive                                # 训练后逐行输入续写
python run.py transformer --prompt "ROMEO:" --prompt "JULIET:"       # 多个 prompt
python run.py transformer --interactive --temperature 0.8 --top-k 30 # 交互 + 采样参数
```

也可以安装为包，使用 `llm-lab` 命令：

```powershell
pip install -e ".[dev]"
llm-lab transformer --steps 2000
```

## 设备选择（NPU / CUDA / CPU）

`--device` 支持 `auto`（默认）、`npu`、`cuda`、`cpu`：

```powershell
python run.py transformer --device npu --steps 2000   # 昇腾 NPU
python run.py transformer --device cuda               # NVIDIA GPU
python run.py char-lstm --device cpu --steps 200      # CPU 调试
```

`auto` 按 **NPU > CUDA > CPU** 选择。NPU 需要安装与 torch 版本匹配的 `torch_npu`，未安装时自动回落到 CUDA/CPU：

```powershell
pip install torch-npu
```

## 预训练权重

训练结束会自动保存权重，默认 `checkpoints/char_lstm.pt` 与 `checkpoints/transformer.pt`，可用 `--checkpoint` 指定路径。

```powershell
python run.py transformer --steps 2000                                   # 训练并保存
python run.py transformer --steps 0 --load checkpoints/transformer.pt    # 加载后只生成
python run.py transformer --steps 500 --load checkpoints/transformer.pt  # 继续训练
```

Transformer 的权重内含 BPE merges 与模型配置，加载时无需再传 `--vocab-size` / `--n-layer` 等参数（57M 参数模型约 228MB）。字符级权重会校验字符词表，换语料时需保持 `--corpus` 一致。

## 三个命令

| 命令 | 语料 | 词表 | 主干 | 规模 |
| --- | --- | --- | --- | --- |
| `verify` | 正弦序列 | — | 手写 LSTM | 2 万参数 |
| `char-lstm` | 内置古诗词（约 1300 字符） | 字符级 | 2 层 LSTM | 0.62M 参数 |
| `transformer` | TinyShakespeare（1.1MB） | 16384 BPE | 12 层 GPT | 约 57M 参数 |

`transformer` 常用参数：

```powershell
python run.py transformer --data my.txt      # 换任意 UTF-8 语料（中文也可）
python run.py transformer --corpus nltk-words  # 用 NLTK 英文词表训练
python run.py transformer --retrain-bpe      # 重新训练分词器
python run.py transformer --prompt "ROMEO:" --interactive  # 自定义输入 + 交互续写
python run.py transformer --n-layer 6 --n-head 6 --n-embd 384 --steps 1000   # 小模型快速试跑
```

## NLTK 英文词库

`data/nltk_words.txt` 来自 `nltk.corpus.words`，共 235,886 个英文词，每行一个。

nltk_data 的语料 zip 存放在 Git LFS 上，官方 `nltk.download()` 在部分网络下会被 GitHub 连接重置；因此该词表通过 HuggingFace 镜像 `nltk-data-hub/words` 导出为纯文本。运行时只读取 txt，不依赖 nltk / pandas / pyarrow。

两种用法：

- 作为训练语料：`--corpus nltk-words`（`char-lstm` 与 `transformer` 均支持）
- 作为分词器词表：`--retrain-bpe --bpe-extra nltk-words`，重建 BPE 时先喂词表，让高频英文单词优先被合并成整词 token

## 训练与生成流程

字符级 LSTM：

1. 字符级分词并构建词表（`CharCorpus`）
2. 随机滑窗采样 `(x, y)`，`y` 是 `x` 右移一位
3. teacher forcing 交叉熵训练，梯度裁剪防梯度爆炸
4. 携带 `(h, c)` 状态逐字符 temperature / top-k 采样续写

Transformer：

1. byte-level BPE 分词（有缓存直接加载，否则现场训练）
2. 按 19:1 切分训练/验证集
3. 线性 warmup + 余弦退火，AdamW + 梯度裁剪
4. 截断到 `block_size` 的上下文自回归生成

## 测试

```powershell
pytest
```

测试覆盖：手写 LSTM 与 `nn.LSTM` 数值一致、字符词表编解码、BPE 编解码与持久化、Transformer 前向与生成、权重保存/加载、设备解析、命令行输入。

## 说明

- `data/` 中的语料缺失时，`transformer` 会自动下载 TinyShakespeare；下载失败则回落到内置小语料。
- 训练使用 `tqdm` 进度条；未安装时自动回退到按间隔打印日志。
- 在上述环境验证通过：Python 3.14.3、torch 2.11.0（CUDA 可用）。
- 在 Windows + cuDNN 下，`nn.LSTM` 可能在进程退出时崩溃，`char-lstm` 命令会自动禁用 cuDNN 规避。
