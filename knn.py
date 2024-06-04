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
