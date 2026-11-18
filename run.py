# -*- coding: utf-8 -*-
"""无需安装即可运行的统一入口。

用法：
    python run.py verify
    python run.py char-lstm --steps 2000
    python run.py transformer --steps 2000
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from llm_lab.cli.main import main  # noqa: E402

if __name__ == "__main__":
    main()
