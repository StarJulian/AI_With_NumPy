
# K-Means 聚类算法

K-Means 是一种常见的非监督学习算法，广泛应用于数据聚类分析。本文将介绍 K-Means 算法的基本概念、实现步骤以及 Python 代码示例。

## 基本概念

K-Means 聚类算法的目标是将数据集分成 \( K \) 个簇，每个簇由具有相似特征的数据点组成。算法通过迭代的方式找到数据点之间的最佳聚类方式，使得同一簇内的数据点尽可能相似，不同簇之间的数据点尽可能不同。

## 算法步骤

1. **选择簇的数量 \( K \)**：初始化 \( K \) 个簇中心（质心）。
2. **分配数据点**：将每个数据点分配到距离最近的簇中心。
3. **更新簇中心**：重新计算每个簇的质心，即所有分配到该簇的数据点的平均值。
4. **重复步骤 2 和 3**：直到簇中心不再发生变化或达到最大迭代次数。

## 距离度量

在 K-Means 算法中，通常使用欧氏距离（Euclidean Distance）来度量数据点之间的距离。欧氏距离的计算公式如下：

\[ d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2} \]

其中，\( x \) 和 \( y \) 分别为两个样本的特征向量。

## 实现代码

下面是一个使用 `numpy` 实现的 K-Means 聚类算法的示例代码：

```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        """
        训练KMeans聚类模型，找到K个簇。
        
        参数：
        - X: 输入数据，形状 (num_samples, num_features)
        """
        # 随机初始化簇中心
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # 分配每个样本到最近的簇中心
            labels = self._assign_clusters(X)
            # 计算新的簇中心
            new_centroids = self._calculate_centroids(X, labels)
            # 检查簇中心是否发生变化
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        """
        分配每个样本到最近的簇中心。
        
        参数：
        - X: 输入数据，形状 (num_samples, num_features)
        
        返回值：
        - labels: 每个样本的簇标签，形状 (num_samples,)
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X, labels):
        """
        计算每个簇的新簇中心。
        
        参数：
        - X: 输入数据，形状 (num_samples, num_features)
        - labels: 每个样本的簇标签，形状 (num_samples,)
        
        返回值：
        - new_centroids: 新的簇中心，形状 (k, num_features)
        """
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def predict(self, X):
        """
        预测每个样本所属的簇。
        
        参数：
        - X: 输入数据，形状 (num_samples, num_features)
        
        返回值：
        - labels: 每个样本的簇标签，形状 (num_samples,)
        """
        return self._assign_clusters(X)

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    X = np.array([
        [1, 2], [1, 4], [1, 0],
        [10, 2], [10, 4], [10, 0]
    ])

    # 创建KMeans实例
    kmeans = KMeans(k=2)
    kmeans.fit(X)
    predictions = kmeans.predict(X)

    print("簇中心:\n", kmeans.centroids)
    print("预测簇标签:", predictions)
```

## 代码解释

1. **初始化**：
   - `__init__` 方法初始化 KMeans 聚类模型，设置簇的数量 \( K \) 和最大迭代次数。

2. **训练模型**：
   - `fit` 方法训练模型，找到 \( K \) 个簇中心。
   - 随机初始化簇中心。
   - 迭代执行分配数据点和更新簇中心的步骤，直到簇中心不再变化或达到最大迭代次数。

3. **分配数据点**：
   - `_assign_clusters` 方法计算每个数据点到所有簇中心的距离，并分配到最近的簇中心。

4. **更新簇中心**：
   - `_calculate_centroids` 方法计算每个簇的新簇中心，即所有分配到该簇的数据点的平均值。

5. **预测**：
   - `predict` 方法根据训练好的模型预测每个数据点所属的簇。

6. **示例用法**：
   - 创建示例数据。
   - 实例化 KMeans 聚类模型，并设置 \( K \) 值为 2。
   - 调用 `fit` 方法训练模型。
   - 调用 `predict` 方法对数据进行预测，并输出簇中心和预测簇标签。

## 超参数选择

- \( K \) 值的选择直接影响聚类结果，可以通过肘部法则（Elbow Method）或轮廓系数（Silhouette Score）来选择最优的 \( K \) 值。
- 最大迭代次数可以根据数据集的规模和算法的收敛速度进行设置。

## 优缺点

### 优点

- 简单直观，易于实现和理解。
- 计算效率高，适用于大规模数据集。
- 对于球状簇效果较好。

### 缺点

- 对初始簇中心敏感，可能陷入局部最优解。
- 需要提前指定簇的数量 \( K \)。
- 对非球状簇和不同大小的簇效果较差。

## 总结

K-Means 是一种经典的聚类算法，适用于将数据集划分成 \( K \) 个簇。尽管其简单性和计算效率使其在许多应用中表现良好，但在处理复杂数据分布时，K-Means 的效果可能不如其他高级聚类算法。通过合理选择 \( K \) 值和初始簇中心，K-Means 可以在许多实际问题中取得令人满意的效果。
