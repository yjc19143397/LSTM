# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 风格。

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
