# AI_From_Scratch

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
AI_From_Scratch/
├── numpy_impl/                    # NumPy 实现版本
│   ├── ML/                       # 机器学习算法
│   │   ├── linear_regression.py   # 线性回归
│   │   ├── logistic_regression.py # 逻辑回归
│   │   ├── decision_tree.py      # 决策树
│   │   ├── random_forest.py      # 随机森林
│   │   ├── svm.py                # 支持向量机
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
│   └── DL/                       # 深度学习组件
│       ├── activations.py        # 激活函数
│       ├── layers.py             # 网络层
│       ├── losses_dl.py          # 深度学习损失函数
│       ├── optimizer.py          # 优化器
│       └── neural_network.py     # 神经网络
│
├── torch_impl/                   # PyTorch 实现版本
│   ├── ML/                       # PyTorch 机器学习
│   ├── DL/                      # PyTorch 深度学习
│   ├── NLP/                     # 自然语言处理
│   ├── CV/                      # 计算机视觉
│   └── ASR/                     # 自动语音识别
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
# NumPy 线性回归
python numpy_impl/ML/linear_regression.py

# PyTorch MLP
python torch_impl/ML/training_framework_torch.py
```

---

## 📚 算法列表

### 机器学习算法

| 算法 | NumPy | PyTorch | 描述 |
|------|-------|---------|------|
| 线性回归 | ✅ | ✅ | 连续值预测 |
| 逻辑回归 | ✅ | ✅ | 二分类/多分类 |
| 决策树 | ✅ | - | CART 算法 |
| 随机森林 | ✅ | - | 集成学习 |
| SVM | ✅ | - | 支持向量机 |
| 朴素贝叶斯 | ✅ | - | 概率分类 |
| K-Means | ✅ | - | 聚类分析 |
| KNN | ✅ | - | K-最近邻 |
| 马尔可夫链 | ✅ | - | 时序建模 |
| 隐马尔可夫 | ✅ | - | HMM |
| PCA | ✅ | - | 降维 |
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
| 优化器 | ✅ | ✅ |

### NLP/CV/ASR

- **NLP**: TextCNN, Word2Vec, Seq2Seq, Transformer, BERT
- **CV**: LeNet, AlexNet, VGGNet, ResNet, MobileNet, YOLO, U-Net
- **ASR**: Wav2Vec, CTC, Deep Speech, Attention ASR

---

## 📄 许可证

MIT License - 仅供学习使用，禁止商业用途
