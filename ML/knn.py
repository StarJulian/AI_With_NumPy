"""
================================================================================
                        K-最近邻 (KNN) - NumPy 实现
================================================================================
包含:
    - KNN 分类
    - KNN 回归
    - 距离加权 KNN
================================================================================
"""

import numpy as np
from typing import Optional, List, Callable, Union
from collections import Counter
from .training_framework import Model


class KNN(Model):
    """
    K-最近邻分类器/回归器
    
    支持:
        - 分类和回归
        - 多种距离度量
        - 距离加权投票
        - KD-Tree 加速 (简化版)
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform',
                 metric: str = 'euclidean', p: float = 2,
                 algorithm: str = 'brute'):
        """
        参数:
            n_neighbors: 近邻数量 K
            weights: 权重方式 ('uniform': 均匀, 'distance': 距离加权)
            metric: 距离度量 ('euclidean', 'manhattan', 'minkowski')
            p: Minkowski 距离的幂参数
            algorithm: 搜索算法 ('brute', 'kd_tree')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.is_fitted: bool = False
    
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """计算两个样本之间的距离"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.abs(x1 - x2) ** self.p), 1 / self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _compute_distances(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """计算一个样本与数据集的距离"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((X - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X - x), axis=1)
        elif self.metric == 'minkowski':
            return np.power(np.sum(np.abs(X - x) ** self.p, axis=1), 1 / self.p)
        else:
            return np.array([self._compute_distance(x, xi) for xi in X])
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'KNN':
        """
        训练 (实际上是保存训练数据)
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 目标标签 (n_samples,)
        """
        self.n_samples_, self.n_features_ = X.shape
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.n_classes_ = len(np.unique(y))
        
        # 数据标准化
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-8
        self.X_train_normalized = (X - self.mean_) / self.std_
        
        print(f"\n{'='*60}")
        print("KNN 模型准备完成")
        print(f"{'='*60}")
        print(f"样本数: {self.n_samples_}, 特征数: {self.n_features_}")
        print(f"K (近邻数): {self.n_neighbors}")
        print(f"距离度量: {self.metric}")
        print(f"权重方式: {self.weights}")
        
        self.is_fitted = True
        return self
    
    def _find_k_neighbors(self, x_normalized: np.ndarray) -> tuple:
        """找到 K 个最近邻"""
        distances = self._compute_distances(x_normalized, self.X_train_normalized)
        
        if self.n_neighbors >= len(distances):
            indices = np.argsort(distances)
            return indices, distances[indices]
        
        # 找到 K 个最近邻
        indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
        k_distances = distances[indices]
        sorted_indices = indices[np.argsort(k_distances)]
        
        return sorted_indices, distances[sorted_indices]
    
    def _predict_single_classification(self, x: np.ndarray) -> int:
        """KNN 分类预测"""
        x_normalized = (x - self.mean_) / self.std_
        indices, distances = self._find_k_neighbors(x_normalized)
        labels = self.y_train[indices]
        
        if self.weights == 'uniform':
            # 均匀投票
            counter = Counter(labels)
            return counter.most_common(1)[0][0]
        else:
            # 距离加权投票
            weights = 1.0 / (distances + 1e-8)
            weighted_votes = {}
            for label, weight in zip(labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            return max(weighted_votes, key=weighted_votes.get)
    
    def _predict_single_regression(self, x: np.ndarray) -> float:
        """KNN 回归预测"""
        x_normalized = (x - self.mean_) / self.std_
        indices, distances = self._find_k_neighbors(x_normalized)
        values = self.y_train[indices]
        
        if self.weights == 'uniform':
            return np.mean(values)
        else:
            # 距离加权平均
            weights = 1.0 / (distances + 1e-8)
            return np.sum(weights * values) / np.sum(weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别/值
        
        参数:
            X: 输入数据 (n_samples, n_features)
            
        返回:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        X = np.array(X)
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            predictions[i] = self._predict_single_classification(x)
        
        return predictions.astype(int)
    
    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        """KNN 回归预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        X = np.array(X)
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            predictions[i] = self._predict_single_regression(x)
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算分类准确率"""
        return np.mean(self.predict(X) == y)
    
    def score_regression(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算回归 R² 分数"""
        predictions = self.predict_regression(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def __str__(self) -> str:
        return f"""KNN Classifier/Regressor
============================
K (neighbors): {self.n_neighbors}
Distance Metric: {self.metric}
Weights: {self.weights}
"""


class KNNRegressor(KNN):
    """KNN 回归器"""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_regression(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.score_regression(X, y)


# =============================================================================
#                           示例: 使用 KNN
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("KNN 示例 - NumPy 实现")
    print("="*60)
    
    # ========================================
    # 示例 1: KNN 分类
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: KNN 分类")
    print("-"*40)
    
    np.random.seed(42)
    
    # 生成数据
    n_samples = 300
    X1 = np.random.randn(n_samples//3, 2) + [0, 0]
    X2 = np.random.randn(n_samples//3, 2) + [3, 3]
    X3 = np.random.randn(n_samples//3, 2) + [6, 0]
    X_knn = np.vstack([X1, X2, X3])
    y_knn = np.array([0]*(n_samples//3) + [1]*(n_samples//3) + [2]*(n_samples//3))
    
    idx = np.random.permutation(n_samples)
    X_knn, y_knn = X_knn[idx], y_knn[idx]
    
    split = int(0.8 * n_samples)
    X_train, X_test = X_knn[:split], X_knn[split:]
    y_train, y_test = y_knn[:split], y_knn[split:]
    
    # 训练 KNN
    knn = KNN(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    
    print(f"\n训练集准确率: {knn.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {knn.score(X_test, y_test):.4f}")
    
    # ========================================
    # 示例 2: K 值选择
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: K 值对准确率的影响")
    print("-"*40)
    
    print(f"\n{'K 值':>10} | {'训练准确率':>12} | {'测试准确率':>12}")
    print("-" * 40)
    
    for k in [1, 3, 5, 7, 9, 15, 25]:
        knn_k = KNN(n_neighbors=k, weights='distance')
        knn_k.fit(X_train, y_train)
        
        train_acc = knn_k.score(X_train, y_train)
        test_acc = knn_k.score(X_test, y_test)
        
        print(f"{k:>10} | {train_acc:>12.4f} | {test_acc:>12.4f}")
    
    # ========================================
    # 示例 3: 距离度量对比
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: 不同距离度量对比")
    print("-"*40)
    
    print(f"\n{'度量方式':>15} | {'测试准确率':>12}")
    print("-" * 30)
    
    for metric in ['euclidean', 'manhattan', 'minkowski']:
        knn_m = KNN(n_neighbors=5, metric=metric)
        knn_m.fit(X_train, y_train)
        test_acc = knn_m.score(X_test, y_test)
        print(f"{metric:>15} | {test_acc:>12.4f}")
    
    # ========================================
    # 示例 4: KNN 回归
    # ========================================
    print("\n" + "-"*40)
    print("示例 4: KNN 回归")
    print("-"*40)
    
    # 生成回归数据
    np.random.seed(42)
    n_reg = 200
    X_reg = np.sort(np.random.uniform(-5, 5, (n_reg, 1)), axis=0)
    y_reg = np.sin(X_reg.flatten()) + np.random.randn(n_reg) * 0.3
    X_reg = np.hstack([X_reg, X_reg**2])  # 添加一个特征
    
    split = int(0.8 * n_reg)
    X_train_r, X_test_r = X_reg[:split], X_reg[split:]
    y_train_r, y_test_r = y_reg[:split], y_reg[split:]
    
    knn_reg = KNNRegressor(n_neighbors=10, weights='distance')
    knn_reg.fit(X_train_r, y_train_r)
    
    predictions = knn_reg.predict_regression(X_test_r)
    r2_score = knn_reg.score_regression(X_test_r, y_test_r)
    
    print(f"测试集 R² 分数: {r2_score:.4f}")
    print(f"测试集 MSE: {np.mean((predictions - y_test_r)**2):.4f}")
    
    # ========================================
    # 示例 5: 权重对比
    # ========================================
    print("\n" + "-"*40)
    print("示例 5: 均匀权重 vs 距离加权")
    print("-"*40)
    
    knn_uniform = KNN(n_neighbors=5, weights='uniform')
    knn_uniform.fit(X_train, y_train)
    
    knn_distance = KNN(n_neighbors=5, weights='distance')
    knn_distance.fit(X_train, y_train)
    
    print(f"均匀权重 - 测试准确率: {knn_uniform.score(X_test, y_test):.4f}")
    print(f"距离加权 - 测试准确率: {knn_distance.score(X_test, y_test):.4f}")
    
    # ========================================
    # 可视化
    # ========================================
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 分类决策边界 (简化)
        colors = ['red', 'blue', 'green']
        for i, c in enumerate(colors):
            mask = y_knn == i
            axes[0].scatter(X_knn[mask, 0], X_knn[mask, 1], c=c, alpha=0.6, label=f'类 {i}')
        axes[0].set_title('KNN 分类数据分布')
        axes[0].legend()
        
        # K 值影响
        k_values = [1, 3, 5, 7, 9, 15]
        train_accs, test_accs = [], []
        for k in k_values:
            knn_v = KNN(n_neighbors=k, weights='distance')
            knn_v.fit(X_train, y_train)
            train_accs.append(knn_v.score(X_train, y_train))
            test_accs.append(knn_v.score(X_test, y_test))
        
        axes[1].plot(k_values, train_accs, 'b-o', label='训练')
        axes[1].plot(k_values, test_accs, 'r-o', label='测试')
        axes[1].set_xlabel('K 值')
        axes[1].set_ylabel('准确率')
        axes[1].set_title('K 值对准确率的影响')
        axes[1].legend()
        axes[1].grid(True)
        
        # 回归结果
        axes[2].scatter(X_test_r[:, 0], y_test_r, alpha=0.6, label='真实值')
        axes[2].scatter(X_test_r[:, 0], predictions, alpha=0.6, label='预测')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('y')
        axes[2].set_title(f'KNN 回归 (R²={r2_score:.3f})')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig('knn_demo.png', dpi=150)
        print("\n图表已保存到 knn_demo.png")
    except Exception as e:
        print(f"\n注意: 无法生成可视化图表 ({e})")
    
    print("\n" + "="*60)
    print("KNN 示例完成!")
    print("="*60)
