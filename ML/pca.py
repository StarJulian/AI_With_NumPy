"""
================================================================================
                        PCA 主成分分析 - NumPy 实现
================================================================================
"""

import numpy as np
from typing import Optional, Tuple
from .training_framework import Model


class PCA(Model):
    """
    主成分分析 (PCA)
    
    用于降维和特征提取
    """
    
    def __init__(self, n_components: int = 2):
        """
        参数:
            n_components: 主成分数量
        """
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'PCA':
        """
        训练 PCA
        
        参数:
            X: 输入数据 (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        print(f"\n{'='*60}")
        print("训练 PCA")
        print(f"{'='*60}")
        print(f"样本数: {n_samples}, 特征数: {n_features}")
        print(f"主成分数: {self.n_components}")
        
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 按特征值排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 选择主成分
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
        self.is_fitted = True
        
        print(f"\n各主成分解释的方差比例:")
        for i, ratio in enumerate(self.explained_variance_ratio_):
            print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        print(f"  总计: {np.sum(self.explained_variance_ratio_):.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """降维转换"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """训练并转换"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """逆转换 (重构)"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return np.dot(X_transformed, self.components_) + self.mean_
    
    def get_params(self) -> dict:
        return {
            'n_components': self.n_components,
            'components': self.components_,
            'explained_variance_ratio': self.explained_variance_ratio_
        }


if __name__ == "__main__":
    print("="*60)
    print("PCA 示例")
    print("="*60)
    
    np.random.seed(42)
    
    # 生成高维数据
    n_samples = 200
    n_features = 10
    
    # 创建相关特征
    X = np.random.randn(n_samples, 3)  # 原始3维
    X[:, 1] = X[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2  # 相关
    X[:, 2] = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1
    
    # 添加噪声
    noise = np.random.randn(n_samples, n_features - 3) * 0.5
    X = np.hstack([X, noise])
    
    print(f"\n原始数据形状: {X.shape}")
    
    # PCA
    for n_comp in [2, 3, 5]:
        pca = PCA(n_components=n_comp)
        X_transformed = pca.fit_transform(X)
        
        print(f"\n降维到 {n_comp} 维:")
        print(f"  转换后形状: {X_transformed.shape}")
        print(f"  累计解释方差: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # 重构误差
    pca = PCA(n_components=3)
    pca.fit(X)
    X_reconstructed = pca.inverse_transform(pca.transform(X))
    mse = np.mean((X - X_reconstructed) ** 2)
    print(f"\n重构误差 (MSE): {mse:.6f}")
    
    print("\n" + "="*60)
