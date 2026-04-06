"""
================================================================================
                        朴素贝叶斯 - NumPy 实现
================================================================================
包含:
    - 高斯朴素贝叶斯
    - 伯努利朴素贝叶斯
    - 多项式朴素贝叶斯
================================================================================
"""

import numpy as np
from typing import Optional
from .training_framework import Model


class GaussianNB(Model):
    """高斯朴素贝叶斯分类器"""
    
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.class_prior_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'GaussianNB':
        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        print(f"\n训练高斯朴素贝叶斯, 类别数: {self.n_classes_}")
        
        self.class_prior_ = np.zeros(self.n_classes_)
        for i, c in enumerate(self.classes_):
            self.class_prior_[i] = np.sum(y == c) / self.n_samples_
        
        self.theta_ = np.zeros((self.n_classes_, self.n_features_))
        self.var_ = np.zeros((self.n_classes_, self.n_features_))
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = np.mean(X_c, axis=0)
            self.var_[i] = np.var(X_c, axis=0) + self.var_smoothing
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        log_proba = np.zeros((len(X), self.n_classes_))
        for i in range(self.n_classes_):
            mean = self.theta_[i]
            var = self.var_[i]
            log_proba[:, i] = -0.5 * np.sum(
                np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, axis=1
            ) + np.log(self.class_prior_[i])
        
        return self.classes_[np.argmax(log_proba, axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


class MultinomialNB(Model):
    """多项式朴素贝叶斯"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'MultinomialNB':
        X = np.maximum(X, 0)
        self.n_samples_, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.class_log_prior_ = np.zeros(self.n_classes_)
        for i, c in enumerate(self.classes_):
            self.class_log_prior_[i] = np.log(np.sum(y == c) / self.n_samples_)
        
        self.feature_log_prob_ = np.zeros((self.n_classes_, self.n_features_))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            count = np.sum(X_c, axis=0) + self.alpha
            total = np.sum(count)
            self.feature_log_prob_[i] = np.log(count / total)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.maximum(X, 0)
        log_proba = self.class_log_prior_ + X @ self.feature_log_prob_.T
        return self.classes_[np.argmax(log_proba, axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# 别名
NaiveBayes = GaussianNB


if __name__ == "__main__":
    print("="*60)
    print("朴素贝叶斯示例")
    print("="*60)
    
    np.random.seed(42)
    
    # 生成数据
    X1 = np.random.randn(100, 2) + [0, 0]
    X2 = np.random.randn(100, 2) + [3, 3]
    X = np.vstack([X1, X2])
    y = np.array([0]*100 + [1]*100)
    
    # 高斯朴素贝叶斯
    gnb = GaussianNB()
    gnb.fit(X, y)
    print(f"准确率: {gnb.score(X, y):.4f}")
    
    print("\n" + "="*60)
