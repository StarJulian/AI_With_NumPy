# ML_From_Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.22+-orange.svg)

**使用 NumPy 从零实现机器学习算法**

[English](./README.md) | [中文](./README_zh.md)

</div>

---

## 项目简介

一个全面的机器学习算法库，使用纯 NumPy 从零实现。每个算法都包含：
- **训练代码**: 完整的梯度下降、坐标下降或 EM 算法实现
- **推理代码**: 支持回归和分类的预测方法
- **Demo 示例**: 可运行的示例代码

> ⚠️ **免责声明**: 仅供学习目的使用，禁止商业用途。

---

## 项目结构

```
ML_From_Scratch/
├── __init__.py                # 包入口
├── training_framework.py      # 训练基类
├── losses.py                  # 损失函数
│
├── linear_regression.py       # 线性回归
├── logistic_regression.py     # 逻辑回归
├── decision_tree.py           # 决策树 (ID3/CART)
├── random_forest.py            # 随机森林
├── svm.py                     # 支持向量机
├── naive_bayes.py             # 朴素贝叶斯
├── kmeans.py                  # K-Means 聚类
├── knn.py                     # K-最近邻
├── markov.py                  # 马尔可夫链
├── hidden_markov.py           # 隐马尔可夫模型
├── pca.py                     # 主成分分析
├── adaboost.py                # AdaBoost
│
├── LICENSE                    # MIT 许可证
└── README_zh.md               # 中文文档
```

---

## 快速开始

### 环境要求
- Python 3.9+
- NumPy 1.22+

```bash
pip install numpy matplotlib
```

### 运行示例

```bash
# 线性回归
python ML/linear_regression.py

# 逻辑回归
python ML/logistic_regression.py

# 决策树
python ML/decision_tree.py

# K-Means
python ML/kmeans.py
```

---

## 算法列表

| 算法 | 文件 | 类型 | 方法 |
|------|------|------|------|
| 线性回归 | `linear_regression.py` | 回归 | 梯度下降 / 正规方程 |
| 逻辑回归 | `logistic_regression.py` | 分类 | 梯度下降 |
| 决策树 | `decision_tree.py` | 分类 | ID3 / CART |
| 随机森林 | `random_forest.py` | 分类 | 集成学习 |
| SVM | `svm.py` | 分类 | SMO |
| 朴素贝叶斯 | `naive_bayes.py` | 分类 | 贝叶斯 |
| K-Means | `kmeans.py` | 聚类 | Lloyd 算法 |
| KNN | `knn.py` | 分类 | 距离加权 |
| 马尔可夫链 | `markov.py` | 序列 | 转移矩阵 |
| 隐马尔可夫 | `hidden_markov.py` | 序列 | Baum-Welch / Viterbi |
| PCA | `pca.py` | 降维 | SVD |
| AdaBoost | `adaboost.py` | 集成 | Boosting |

---

## 使用示例

```python
from ML import LinearRegression
import numpy as np

# 训练数据
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# 训练模型
model = LinearRegression(input_dim=1, method='gd')
model.fit(X, y, learning_rate=0.01, max_iter=1000)

# 预测
X_test = np.array([[6], [7]])
y_pred = model.predict(X_test)
print(f"预测结果: {y_pred}")  # [12, 14]
```

---

## 许可证

MIT License - 仅供学习使用
