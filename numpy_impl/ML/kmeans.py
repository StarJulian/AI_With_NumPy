"""
================================================================================
                        K-Means 聚类 - NumPy 实现
================================================================================
包含:
    - 标准 K-Means
    - Mini-Batch K-Means
    - K-Means++ 初始化
================================================================================
"""

import numpy as np
from typing import Optional, Tuple, List
from .training_framework import Model


class KMeans(Model):
    """
    K-Means 聚类算法
    
    支持:
        - K-Means++ 初始化
        - Mini-Batch 变体
        - 多种收敛策略
    """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 tol: float = 1e-4, init: str = 'kmeans++',
                 n_init: int = 10, batch_size: int = 100,
                 random_state: Optional[int] = None):
        """
        参数:
            n_clusters: 簇的数量
            max_iter: 最大迭代次数
            tol: 收敛阈值
            init: 初始化方法 ('random', 'kmeans++')
            n_init: 初始化次数 (选择最优结果)
            batch_size: Mini-Batch 大小 (batch_size < n_samples 时启用 mini-batch)
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.batch_size = batch_size
        self.random_state = random_state
        
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0  # 簇内平方和
        self.n_iter_: int = 0
        self.is_fitted: bool = False
    
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """初始化聚类中心"""
        if self.init == 'random':
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            return X[indices].copy()
        
        elif self.init == 'kmeans++':
            return self._kmeans_plus_plus_init(X)
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
    
    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means++ 初始化
        
        策略: 选择距离已有中心最远的点作为下一个中心
        """
        n_samples = len(X)
        
        # 随机选择第一个中心
        centroids = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # 计算每个点到最近中心的距离
            distances = np.zeros(n_samples)
            for x in X:
                min_dist = float('inf')
                for c in centroids:
                    dist = np.sum((x - c) ** 2)
                    min_dist = min(min_dist, dist)
                distances[np.where(np.all(X == x, axis=1))[0]] = min_dist
            
            # 根据距离概率分布选择下一个中心
            probabilities = distances / distances.sum()
            next_centroid_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_centroid_idx])
        
        return np.array(centroids)
    
    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """计算每个点到各个中心的距离"""
        n_samples = len(X)
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)
        
        return distances
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """分配每个点到最近的聚类中心"""
        distances = self._compute_distances(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """更新聚类中心"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                new_centroids[i] = np.mean(X[mask], axis=0)
            else:
                # 如果有空簇，随机选择一个点作为新中心
                new_centroids[i] = X[np.random.randint(len(X))]
        
        return new_centroids
    
    def _compute_inertia(self, X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
        """计算簇内平方和 (Inertia)"""
        inertia = 0.0
        for i in range(self.n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                inertia += np.sum((X[mask] - centroids[i]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'KMeans':
        """
        训练 K-Means 模型
        
        参数:
            X: 输入数据 (n_samples, n_features)
            y: 无监督学习中通常为 None
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_samples_, self.n_features_ = X.shape
        
        print(f"\n{'='*60}")
        print("训练 K-Means 聚类")
        print(f"{'='*60}")
        print(f"样本数: {self.n_samples_}, 特征数: {self.n_features_}")
        print(f"簇数: {self.n_clusters}")
        print(f"初始化方法: {self.init}")
        
        use_minibatch = self.batch_size < self.n_samples_
        if use_minibatch:
            print(f"Mini-Batch 模式: batch_size = {self.batch_size}")
        
        best_centroids = None
        best_inertia = float('inf')
        best_labels = None
        best_n_iter = 0
        
        for init_run in range(self.n_init):
            # 初始化聚类中心
            centroids = self._init_centroids(X)
            
            for iteration in range(self.max_iter):
                if use_minibatch:
                    # Mini-Batch K-Means
                    batch_indices = np.random.choice(self.n_samples_, self.batch_size, replace=False)
                    X_batch = X[batch_indices]
                else:
                    X_batch = X
                
                # 分配簇
                labels = self._assign_clusters(X_batch, centroids)
                
                # 更新中心
                new_centroids = self._update_centroids(X_batch, labels)
                
                # 检查收敛
                shift = np.sum((new_centroids - centroids) ** 2)
                centroids = new_centroids
                
                if shift < self.tol:
                    break
            
            # 计算 inertia
            all_labels = self._assign_clusters(X, centroids)
            inertia = self._compute_inertia(X, centroids, all_labels)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = all_labels
                best_n_iter = iteration + 1
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.is_fitted = True
        
        print(f"\n训练完成!")
        print(f"迭代次数: {self.n_iter_}")
        print(f"簇内平方和 (Inertia): {self.inertia_:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测簇标签
        
        参数:
            X: 输入数据 (n_samples, n_features)
            
        返回:
            簇标签 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        return self._assign_clusters(X, self.cluster_centers_)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据到簇距离空间
        
        返回:
            每个点到各簇中心的距离 (n_samples, n_clusters)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        return self._compute_distances(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """训练并预测"""
        return self.fit(X).predict(X)
    
    def score(self, X: np.ndarray) -> float:
        """返回负的 inertia (用于 sklearn API 兼容性)"""
        return -self.inertia_
    
    def get_cluster_centers(self) -> np.ndarray:
        """获取聚类中心"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.cluster_centers_.copy()
    
    def __str__(self) -> str:
        return f"""K-Means Clustering
=====================
Number of Clusters: {self.n_clusters}
Iterations: {self.n_iter_}
Inertia: {self.inertia_:.4f}
"""


# =============================================================================
#                           示例: 使用 K-Means
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("K-Means 聚类示例 - NumPy 实现")
    print("="*60)
    
    # ========================================
    # 示例 1: 基本聚类
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: 基本聚类")
    print("-"*40)
    
    np.random.seed(42)
    
    # 生成三个簇的数据
    n_samples = 300
    centers = [[0, 0], [5, 5], [-3, 3]]
    X_cluster = []
    for center in centers:
        X_cluster.append(np.random.randn(n_samples//3, 2) + center)
    X_cluster = np.vstack(X_cluster)
    
    # 打乱
    np.random.shuffle(X_cluster)
    
    print(f"数据点: {len(X_cluster)}")
    
    # 训练 K-Means
    kmeans = KMeans(n_clusters=3, init='kmeans++', random_state=42)
    kmeans.fit(X_cluster)
    
    labels = kmeans.predict(X_cluster)
    centroids = kmeans.get_cluster_centers()
    
    print(f"\n聚类中心:")
    for i, c in enumerate(centroids):
        print(f"  簇 {i}: {c}")
    print(f"\n各簇样本数: {np.bincount(labels)}")
    
    # ========================================
    # 示例 2: 手肘法确定 K
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: 手肘法确定最优 K 值")
    print("-"*40)
    
    inertias = []
    K_range = range(1, 10)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_cluster)
        inertias.append(km.inertia_)
        print(f"K={k}: Inertia={km.inertia_:.4f}")
    
    # ========================================
    # 示例 3: Mini-Batch K-Means 对比
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: Mini-Batch vs 标准 K-Means")
    print("-"*40)
    
    # 大数据集
    np.random.seed(42)
    n_large = 10000
    X_large = np.random.randn(n_large, 2) * 10
    
    import time
    
    # 标准 K-Means
    start = time.time()
    km_full = KMeans(n_clusters=5, batch_size=n_large, random_state=42)
    km_full.fit(X_large)
    time_full = time.time() - start
    
    # Mini-Batch K-Means
    start = time.time()
    km_batch = KMeans(n_clusters=5, batch_size=100, random_state=42)
    km_batch.fit(X_large)
    time_batch = time.time() - start
    
    print(f"\n标准 K-Means: 时间={time_full:.4f}s, Inertia={km_full.inertia_:.4f}")
    print(f"Mini-Batch K-Means: 时间={time_batch:.4f}s, Inertia={km_batch.inertia_:.4f}")
    print(f"加速比: {time_full/time_batch:.2f}x")
    
    # ========================================
    # 示例 4: 图像压缩
    # ========================================
    print("\n" + "-"*40)
    print("示例 4: K-Means 用于图像压缩")
    print("-"*40)
    
    # 模拟图像数据 (简化版)
    np.random.seed(42)
    n_pixels = 500
    X_image = np.random.randint(0, 256, (n_pixels, 3)) / 255.0  # RGB
    
    print(f"原始像素数: {len(X_image)}")
    print(f"原始颜色空间: {256**3:,} 种颜色")
    
    for n_colors in [2, 4, 8, 16, 32]:
        km_img = KMeans(n_clusters=n_colors, random_state=42)
        km_img.fit(X_image)
        
        compressed = km_img.predict(X_image)
        n_used_colors = len(np.unique(compressed))
        
        compression_ratio = n_pixels * 24 / (n_pixels * np.log2(n_colors) + n_colors * 3)
        
        print(f"  {n_colors} 色: 实际使用 {n_used_colors} 色, 压缩比约 {compression_ratio:.2f}x")
    
    # ========================================
    # 可视化
    # ========================================
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 聚类结果
        colors = ['red', 'blue', 'green']
        for i in range(3):
            mask = labels == i
            axes[0].scatter(X_cluster[mask, 0], X_cluster[mask, 1], 
                           c=colors[i], alpha=0.6, label=f'簇 {i}')
        axes[0].scatter(centroids[:, 0], centroids[:, 1], c='black', 
                       marker='x', s=200, linewidths=3, label='中心')
        axes[0].set_title('K-Means 聚类结果')
        axes[0].legend()
        
        # 手肘曲线
        axes[1].plot(K_range, inertias, 'bo-')
        axes[1].set_xlabel('K (簇数)')
        axes[1].set_ylabel('Inertia (簇内平方和)')
        axes[1].set_title('手肘法')
        axes[1].grid(True)
        
        # Mini-Batch 对比
        axes[2].bar(['标准 K-Means', 'Mini-Batch'], [time_full, time_batch])
        axes[2].set_ylabel('时间 (秒)')
        axes[2].set_title('训练时间对比')
        
        plt.tight_layout()
        plt.savefig('kmeans_demo.png', dpi=150)
        print("\n图表已保存到 kmeans_demo.png")
    except Exception as e:
        print(f"\n注意: 无法生成可视化图表 ({e})")
    
    print("\n" + "="*60)
    print("K-Means 示例完成!")
    print("="*60)
