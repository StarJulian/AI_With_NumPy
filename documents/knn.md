
# K-最近邻（KNN）算法

K-最近邻（K-Nearest Neighbors, KNN）是一种简单且直观的监督学习算法，广泛应用于分类和回归任务。本文将介绍KNN算法的基本概念、实现细节以及Python代码示例。

## 基本概念

KNN算法的核心思想是：给定一个测试样本，根据其在特征空间中与训练样本的距离，找到距离最近的K个训练样本（邻居），然后通过这些邻居的标签来决定测试样本的标签。在分类任务中，KNN通过对K个邻居的标签进行投票，选择出现次数最多的标签作为预测结果；在回归任务中，KNN通过对K个邻居的标签进行平均来预测结果。

## 算法步骤

1. **计算距离**：计算测试样本与每个训练样本之间的距离。
2. **选择最近的K个邻居**：根据距离选择K个最近的训练样本。
3. **投票**：在K个最近邻居中，选择出现次数最多的类别作为预测结果。

## 距离度量

在KNN算法中，通常使用欧氏距离（Euclidean Distance）来度量样本之间的距离。欧氏距离的计算公式如下：

\[ d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2} \]

其中，\( x \) 和 \( y \) 分别为两个样本的特征向量。

## 实现代码

下面是一个使用 `numpy` 实现的 KNN 分类器的示例代码：

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        """
        训练KNN分类器，保存训练数据。
        
        参数：
        - X_train: 训练样本特征，形状 (num_samples, num_features)
        - y_train: 训练样本标签，形状 (num_samples,)
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        对测试样本进行预测。
        
        参数：
        - X_test: 测试样本特征，形状 (num_samples, num_features)
        
        返回值：
        - y_pred: 预测标签，形状 (num_samples,)
        """
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        """
        对单个测试样本进行预测。
        
        参数：
        - x: 单个测试样本特征，形状 (num_features,)
        
        返回值：
        - 预测标签
        """
        # 计算所有训练样本与测试样本之间的距离
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # 获取距离最近的k个训练样本的索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取k个最近邻居的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 返回出现次数最多的标签
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2, 3], [3, 5], [8, 8]])

    # 创建KNN实例
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    print("测试样本预测结果:", predictions)
```

## 代码解释

1. **初始化**：
   - `__init__` 方法初始化KNN分类器，并设置K值。

2. **训练模型**：
   - `fit` 方法保存训练样本的特征和标签，供后续预测使用。

3. **预测**：
   - `predict` 方法对一组测试样本进行预测，返回预测标签。
   - `_predict` 方法对单个测试样本进行预测：
     - 计算测试样本与每个训练样本之间的欧氏距离。
     - 找到距离最近的K个训练样本的索引。
     - 获取K个最近邻居的标签。
     - 返回出现次数最多的标签作为预测结果。

4. **示例用法**：
   - 创建示例训练数据和测试数据。
   - 实例化KNN分类器，并设置K值为3。
   - 调用 `fit` 方法训练模型。
   - 调用 `predict` 方法对测试样本进行预测，并输出预测结果。

## 超参数选择

K值是KNN算法的一个关键超参数，其选择会直接影响模型的性能。一般来说，较小的K值会导致模型对噪声敏感，而较大的K值会使模型过于平滑，导致欠拟合。可以通过交叉验证来选择最优的K值。

## 优缺点

### 优点

- 简单直观，易于理解和实现。
- 不需要显式的训练过程，只需保存训练数据。
- 对于小规模数据集效果较好。

### 缺点

- 计算复杂度高，对大规模数据集不适用。
- 对噪声和不相关特征敏感。
- 需要保存所有训练数据，存储开销大。

## 总结

K-最近邻（KNN）是一种经典的机器学习算法，适用于分类和回归任务。尽管其简单性和直观性使其在许多应用中表现良好，但在处理大规模数据集和高维数据时，KNN的计算复杂度和存储需求成为其主要限制因素。通过合理选择K值和使用适当的距离度量，KNN可以在许多实际问题中取得令人满意的效果。