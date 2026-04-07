# AI_With_NumPy

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**使用 NumPy 和 PyTorch 从零实现各种 AI 算法的学习项目**

[English](./README.md) | [中文](./README_zh.md)

</div>

---

## 📖 项目简介

本项目是一个全面的 AI 算法学习仓库，使用纯 NumPy 和 PyTorch 两种框架从零实现各种机器学习、深度学习、NLP、CV 和 ASR 算法。通过阅读和运行这些代码，你可以深入理解算法的底层原理，而不仅仅是调用高层 API。

> ⚠️ **免责声明**: 本项目仅供学习目的使用，禁止商业用途。

---

## 📁 项目结构

```
AI_With_NumPy/
├── numpy_impl/                    # NumPy 实现版本 (从零实现)
│   ├── ML/                       # 机器学习算法
│   │   ├── linear_regression.py   # 线性回归
│   │   ├── logistic_regression.py # 逻辑回归
│   │   ├── decision_tree.py      # 决策树 (ID3/CART)
│   │   ├── random_forest.py      # 随机森林
│   │   ├── svm.py                # 支持向量机 (SMO)
│   │   ├── naive_bayes.py        # 朴素贝叶斯
│   │   ├── kmeans.py             # K-Means 聚类
│   │   ├── knn.py                # K-最近邻
│   │   ├── markov.py             # 马尔可夫链
│   │   ├── hidden_markov.py      # 隐马尔可夫模型
│   │   ├── pca.py                # 主成分分析
│   │   ├── adaboost.py           # AdaBoost
│   │   ├── losses.py             # 损失函数
│   │   └── training_framework.py  # 训练框架
│   │
│   ├── DL/                       # 深度学习组件
│   │   ├── activations.py         # 激活函数
│   │   ├── layers.py              # 网络层
│   │   ├── losses_dl.py           # 深度学习损失函数
│   │   ├── optimizer.py           # 优化器
│   │   └── neural_network.py      # 神经网络
│   │
│   ├── NLP/                      # 自然语言处理
│   │   └── __init__.py           # TextCNN, Word2Vec, N-Gram
│   │
│   ├── CV/                       # 计算机视觉
│   │   └── __init__.py           # LeNet, HOG, 卷积/池化
│   │
│   └── ASR/                      # 自动语音识别
│       └── __init__.py           # MFCC, CTC, SimpleASR
│
├── torch_impl/                    # PyTorch 实现版本
│   ├── ML/                       # PyTorch 机器学习
│   │   ├── linear_regression_torch.py
│   │   ├── logistic_regression_torch.py
│   │   └── training_framework_torch.py
│   │
│   ├── DL/                       # PyTorch 深度学习
│   │
│   ├── NLP/                      # 自然语言处理
│   │   └── nlp_models.py         # TextCNN, Word2Vec, Seq2Seq, Transformer, BERT
│   │
│   ├── CV/                       # 计算机视觉
│   │   └── cv_models.py          # LeNet, AlexNet, VGGNet, ResNet, MobileNet, YOLO, U-Net
│   │
│   └── ASR/                      # 自动语音识别
│       └── __init__.py           # Wav2Vec, CTC, DeepSpeech, AttentionASR
│
├── LICENSE                       # MIT 许可证
├── README.md                     # 英文文档
└── README_zh.md                 # 中文文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- NumPy 1.22+
- PyTorch 2.0+ (可选)

### 安装

```bash
pip install numpy torch matplotlib
```

### 运行示例

```bash
# NumPy 版本
cd numpy_impl/ML
python linear_regression.py     # 线性回归
python logistic_regression.py   # 逻辑回归
python decision_tree.py         # 决策树
python kmeans.py               # K-Means

# PyTorch 版本
cd torch_impl/
python NLP/nlp_models.py       # NLP 模型
python CV/cv_models.py          # CV 模型
python ASR/__init__.py          # ASR 模型
```

---

## 📚 算法列表

### 机器学习算法

| 算法 | NumPy | PyTorch | 描述 |
|------|-------|---------|------|
| 线性回归 | ✅ | ✅ | 梯度下降/正规方程 |
| 逻辑回归 | ✅ | ✅ | 多分类/Sigmoid |
| 决策树 | ✅ | - | ID3/CART |
| 随机森林 | ✅ | - | 集成学习 |
| SVM | ✅ | - | SMO/核函数 |
| 朴素贝叶斯 | ✅ | - | 高斯/多项式 |
| K-Means | ✅ | - | K-Means++ |
| KNN | ✅ | - | 距离加权 |
| 马尔可夫链 | ✅ | - | 时序建模 |
| 隐马尔可夫 | ✅ | - | Baum-Welch/Viterbi |
| PCA | ✅ | - | SVD 分解 |
| AdaBoost | ✅ | - | 提升算法 |

### 深度学习组件

| 组件 | NumPy | PyTorch |
|------|-------|---------|
| 激活函数 | ✅ | ✅ |
| 全连接层 | ✅ | ✅ |
| 卷积层 | ✅ | ✅ |
| 池化层 | ✅ | ✅ |
| Dropout | ✅ | ✅ |
| BatchNorm | ✅ | ✅ |
| LayerNorm | ✅ | ✅ |
| 优化器 | ✅ | ✅ |

### NLP 模型

| 模型 | NumPy | PyTorch | 描述 |
|------|-------|---------|------|
| TextCNN | ✅ | ✅ | 多尺度卷积 |
| Word2Vec | ✅ | ✅ | Skip-Gram/CBOW |
| N-Gram | ✅ | - | 语言模型 |
| Seq2Seq | - | ✅ | + Attention |
| Transformer | - | ✅ | 自注意力 |
| BERT | - | ✅ | 预训练模型 |

### CV 模型

| 模型 | NumPy | PyTorch | 描述 |
|------|-------|---------|------|
| LeNet-5 | ✅ | ✅ | 经典CNN |
| AlexNet | - | ✅ | 2012冠军 |
| VGGNet | - | ✅ | VGG-11/16/19 |
| ResNet | - | ✅ | 残差网络 |
| MobileNetV2 | - | ✅ | 轻量级 |
| YOLOv3 | - | ✅ | 目标检测 |
| U-Net | - | ✅ | 语义分割 |
| HOG | ✅ | - | 特征提取 |

### ASR 模型

| 模型 | NumPy | PyTorch | 描述 |
|------|-------|---------|------|
| MFCC | ✅ | - | 音频特征 |
| SimpleASR | ✅ | - | RNN |
| CTC | ✅ | ✅ | 解码 |
| Wav2Vec | - | ✅ | 自监督 |
| DeepSpeech | - | ✅ | 百度模型 |
| AttentionASR | - | ✅ | 注意力 |

---

## 📄 许可证

MIT License - 仅供学习使用，禁止商业用途
