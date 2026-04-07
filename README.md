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
│   │   ├── linear_regression_torch.py
│   │   ├── logistic_regression_torch.py
│   │   └── training_framework_torch.py
│   │
│   ├── DL/                      # PyTorch 深度学习
│   │   ├── mlp.py              # 多层感知机
│   │   ├── cnn.py              # 卷积神经网络
│   │   ├── rnn.py              # RNN/LSTM
│   │   └── transformer.py       # Transformer
│   │
│   ├── NLP/                     # 自然语言处理
│   │   └── nlp_models.py       # NLP 模型
│   │
│   ├── CV/                      # 计算机视觉
│   │   └── cv_models.py         # CV 模型
│   │
│   └── ASR/                     # 自动语音识别
│       └── asr_models.py        # ASR 模型
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- NumPy 1.22+
- PyTorch 2.0+ (可选，用于 PyTorch 版本)

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/AI_From_Scratch.git
cd AI_From_Scratch

# 安装依赖
pip install numpy torch matplotlib
```

### 运行示例

```bash
# NumPy 线性回归
python numpy_impl/ML/linear_regression.py

# NumPy 逻辑回归
python numpy_impl/ML/logistic_regression.py

# NumPy 决策树
python numpy_impl/ML/decision_tree.py

# PyTorch MLP
python torch_impl/ML/training_framework_torch.py

# PyTorch CNN
python torch_impl/DL/neural_network.py
```

---

## 📚 算法列表

### 机器学习算法 (Machine Learning)

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

### 深度学习组件 (Deep Learning)

| 组件 | NumPy | PyTorch | 描述 |
|------|-------|---------|------|
| 激活函数 | ✅ | ✅ | ReLU/Sigmoid/Tanh/GELU |
| 全连接层 | ✅ | ✅ | Linear |
| 卷积层 | ✅ | ✅ | Conv2D |
| 池化层 | ✅ | ✅ | MaxPool |
| Dropout | ✅ | ✅ | 正则化 |
| BatchNorm | ✅ | ✅ | 批归一化 |
| LayerNorm | ✅ | ✅ | 层归一化 |
| 优化器 | ✅ | ✅ | SGD/Adam/Momentum |

### 自然语言处理 (NLP)

| 模型 | 描述 |
|------|------|
| TextCNN | 卷积文本分类 |
| Word2Vec | 词向量学习 |
| SequenceTagger | 序列标注 (NER/POS) |
| Seq2Seq | 序列到序列 |
| Transformer | Transformer 编码器 |
| BERT | 双向 Transformer |

### 计算机视觉 (CV)

| 模型 | 描述 |
|------|------|
| LeNet-5 | 手写数字识别 |
| AlexNet | ImageNet 竞赛冠军 |
| VGGNet | VGG 架构 |
| ResNet | 残差网络 |
| MobileNet | 轻量级网络 |
| YOLO | 目标检测 |
| U-Net | 语义分割 |

### 自动语音识别 (ASR)

| 模型 | 描述 |
|------|------|
| Wav2Vec | 自监督语音表示 |
| CTC | 连接时序分类 |
| Deep Speech 2 | 端到端 ASR |
| Attention ASR | 注意力机制 ASR |

---

## 📖 学习指南

### 推荐学习路径

1. **入门阶段**
   - 线性回归 → 逻辑回归
   - 激活函数 → 全连接层
   - 反向传播原理

2. **进阶阶段**
   - CNN 图像分类
   - RNN/LSTM 序列建模
   - Word2Vec 词向量

3. **高级阶段**
   - Transformer/Attention
   - ResNet/DenseNet
   - BERT/GPT

4. **专业领域**
   - YOLO 目标检测
   - U-Net 语义分割
   - Wav2Vec/CTC 语音识别

---

## 📝 代码风格

每个算法的实现都遵循以下结构：

```python
class MyAlgorithm:
    def __init__(self, params):
        """初始化"""
        pass
    
    def fit(self, X, y):
        """训练"""
        pass
    
    def predict(self, X):
        """预测"""
        pass
    
    def score(self, X, y):
        """评估"""
        pass

if __name__ == "__main__":
    # 示例代码
    print("算法示例")
```

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

**⚠️ 重要**: 本项目仅供学习目的使用，禁止商业用途。

---

## 🙏 致谢

- NumPy 团队
- PyTorch 团队
- 所有开源贡献者
