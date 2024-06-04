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