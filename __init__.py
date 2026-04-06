"""
================================================================================
                    AI_With_NumPy & PyTorch 学习项目
================================================================================

项目名称: AI_From_Scratch
描述: 使用 NumPy 和 PyTorch 从零实现各种 AI 算法的学习项目
版本: 1.0.0
许可证: MIT License (仅供学习使用，禁止商业用途)

项目结构:
├── numpy_impl/          # NumPy 实现版本
│   ├── ML/              # 机器学习算法
│   ├── DL/              # 深度学习组件
│   ├── NLP/             # 自然语言处理
│   ├── CV/              # 计算机视觉
│   └── ASR/             # 自动语音识别
│
├── torch_impl/          # PyTorch 实现版本
│   ├── ML/              # 机器学习算法
│   ├── DL/              # 深度学习组件
│   ├── NLP/             # 自然语言处理
│   ├── CV/              # 计算机视觉
│   └── ASR/             # 自动语音识别
│
└── LICENSE              # 许可证文件

-----------------------------------------------------------------
⚠️ 免责声明 / Disclaimer
-----------------------------------------------------------------
本项目仅供学习和教育目的使用。

- 本项目中的所有代码和算法实现仅用于学习、研究和教育目的
- 禁止将本项目用于任何商业目的
- 本项目不提供任何明示或暗示的保证
- 使用本项目代码产生的任何后果由使用者自行承担

This project is for learning and educational purposes only.
- All code and algorithm implementations are for learning, research, and education
- Commercial use is strictly prohibited
- No warranty is provided, express or implied
- Users are responsible for any consequences from using this code

================================================================================
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "AI Learning Community"
__license__ = "MIT"

# 项目根目录
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
