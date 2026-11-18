# -*- coding: utf-8 -*-
"""让 tests 直接找到 src/llm_lab，无需先安装包。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
