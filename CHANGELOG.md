# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 风格。

## [0.7.0] - 2026-11-18

### 新增

- 多厂商 NPU / AI 加速设备兼容：`--device` 新增 `mlu`（寒武纪 Cambricon）、`musa`（摩尔线程 Moore Threads）、`dipu`（海光 DCU / 天数智芯 / 壁仞等 DIPU 统一算力层）、`xpu`（Intel XPU GPU）、`mps`（Apple Silicon）；`auto` 按“专用 NPU / 国产算力卡 > CUDA > CPU”优先级自动选择第一个可用设备。
- `config.py` 引入 `_BACKENDS` 注册表与通用检测器：先 import 触发注册，再查 `torch.<attr>.is_available()`；对未注册 `is_available` 的后端（musa / dipu）按 import 成功判定（参照 mmengine），避免漏掉真实硬件。显式指定不可用设备时退出并给出对应安装提示。
- `list_accelerators()`：返回本机实际可用的加速后端列表，供展示与诊断。
- `tests/test_device.py`：用 monkeypatch 模拟各厂商包可用性，覆盖显式解析、`auto` 优先级、报错提示、DIPU 设备解析与 MPS 特殊分支（21 个用例）。

### 变更

- 版本号提升至 0.7.0；README 设备章节改为多厂商优先级矩阵。

## [0.6.0] - 2026-11-18

### 新增

- Intel NPU（AI Boost）推理支持：`--intel-npu` 通过 OpenVINO 把模型固定形状编译到 NPU，仅用于生成阶段，训练仍在 `--device` 上。
- `generate_chars` 与 `GPT.generate` 新增 `forward` 参数，可替换前向计算；字符级新增 `block_size` 参数供固定形状推理使用。
- `tests/test_intel_npu.py`：NPU 前向数值对齐（fp16 容差）、forward 接口测试。

### 变更

- `--device npu` 明确仅指昇腾（torch_npu）；无 torch_npu 时给出提示并退出。
- 版本号提升至 0.6.0，README 增加 Intel NPU 用法。

## [0.5.0] - 2026-11-18

### 新增

- 设备解析 `resolve_device()` 与 `--device auto|npu|cuda|cpu`：`auto` 按 NPU > CUDA > CPU 选择，安装 `torch_npu` 后自动启用昇腾 NPU。
- 预训练权重保存/加载（`checkpoint.py`）：训练结束自动保存，`--checkpoint` 指定路径，`--load` 加载后继续训练，配合 `--steps 0` 可只生成。
- Transformer 权重自含 BPE merges 与模型配置，加载时无需重复传模型参数；字符级权重会校验字符词表。
- `tqdm` 训练进度条，实时显示 loss / ppl / val / lr；未安装时回退到按间隔打印。

### 变更

- 版本号提升至 0.5.0，`requirements.txt` 与 `pyproject.toml` 增加 `tqdm>=4.66`。
- 训练循环与生成函数改为显式传入 `device`，不再依赖模块级常量。

## [0.4.0] - 2026-11-18

### 新增

- `--prompt`（可重复传入）与 `--interactive` 交互式续写，`char-lstm` 与 `transformer` 均支持。
- `--max-new-tokens`、`--temperature`、`--top-k` 可配置生成参数；不传时保持原有默认示例。
- 交互模式对字符级模型的词表外字符给出提示并继续，不中断会话。
- `tests/test_cli_input.py`：参数解析、交互退出与未知字符处理。

### 变更

- 版本号提升至 0.4.0，README 补充自定义输入用法。

### 修复

- `char-lstm` 切换语料后，默认示例 prompt 仍为中文诗句，导致字符词表外字符 KeyError；现默认取当前语料开头作为 prompt。

## [0.3.0] - 2026-11-18

### 新增

- 集成 NLTK 英文词库：`data/nltk_words.txt`，共 235,886 词，来自 `nltk.corpus.words`（经 HuggingFace 镜像导出，运行时不依赖 nltk/pandas/pyarrow）。
- `char-lstm --corpus nltk-words` 与 `transformer --corpus nltk-words`：用词表作为训练语料。
- `transformer --bpe-extra nltk-words`：重建 BPE 时先喂词表，提升英文整词覆盖。
- `tests/test_nltk_corpus.py`：词表加载、去重与非空校验、CLI 语料选择。

### 变更

- 版本号提升至 0.3.0，README 补充 NLTK 词库来源与用法。

## [0.2.0] - 2026-11-18

### 变更

- 将三个平铺脚本重构为 `src/llm_lab` 包，按配置、数据、分词、模型、训练、生成、CLI 分层。
- 模型拆分为 `models/lstm.py`、`models/char_lm.py`、`models/transformer.py`。
- 新增统一命令行入口 `llm-lab` 与免安装运行入口 `run.py`，提供 `verify`、`char-lstm`、`transformer` 三个子命令。
- 语料与分词器缓存统一迁移到 `data/` 目录，修复原先路径与文件位置不一致导致重复下载/重训的问题。
- 训练与生成函数改为显式传入 `device`，便于在 CPU 上测试。

### 新增

- `tests/` 单元测试：LSTM 数值对齐、字符词表、BPE 编解码与持久化、Transformer 前向与生成。
- `pyproject.toml`、`requirements.txt`、`.gitignore`。
- `README.md` 项目结构、用法与流程说明。

### 修复

- 修复语料过小时验证集无法切出训练窗口的问题（`get_batch` 增加长度校验，验证集下限提高到 `block_size + 2`）。

### 移除

- 根目录下的 `lstm.py`、`lstm_lm.py`、`transformer_lm.py`（功能已迁移到包内）。
