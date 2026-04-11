"""
================================================================================
                        AdaBoost - NumPy 实现
================================================================================
"""

import numpy as np
from typing import Optional, List
from .training_framework import Model
from .decision_tree import DecisionTree


class AdaBoost(Model):
    """
    AdaBoost 自适应提升算法
    
    通过组合多个弱分类器构建强分类器
    """
    
    def __init__(self, n_estimators: int = 50, base_estimator: str = 'stump',
                 learning_rate: float = 1.0, random_state: Optional[int] = None):
        """
        参数:
            n_estimators: 弱分类器数量
            base_estimator: 基础分类器类型
            learning_rate: 学习率
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.estimators: List[DecisionTree] = []
        self.estimator_weights: List[float] = []
        self.estimator_errors: List[float] = []
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'AdaBoost':
        """
        训练 AdaBoost
        """
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples = len(X)
        
        # 初始化权重
        sample_weights = np.ones(n_samples) / n_samples
        
        print(f"\n{'='*60}")
        print("训练 AdaBoost")
        print(f"{'='*60}")
        print(f"样本数: {n_samples}, 特征数: {X.shape[1]}")
        print(f"弱分类器数: {self.n_estimators}")
        
        for i in range(self.n_estimators):
            # 训练弱分类器
            estimator = DecisionTree(
                criterion='gini',
                max_depth=1 if self.base_estimator == 'stump' else 3,
                min_samples_split=2,
                random_state=self.random_state + i if self.random_state else None
            )
            estimator.fit(X, y)
            
            # 预测
            predictions = estimator.predict(X)
            
            # 计算误差
            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # 防止数值问题
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # 计算分类器权重
            weight = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # 更新样本权重
            sample_weights *= np.exp(-weight * y * (2 * predictions - 1))
            sample_weights /= np.sum(sample_weights)  # 归一化
            
            self.estimators.append(estimator)
            self.estimator_weights.append(weight)
            self.estimator_errors.append(error)
            
            if (i + 1) % 10 == 0:
                # 手动计算准确率
                weighted_votes = np.zeros((n_samples, 2))
                for j, (est, w) in enumerate(zip(self.estimators, self.estimator_weights)):
                    preds = est.predict(X)
                    weighted_votes[:, 0] += w * (preds == 0)
                    weighted_votes[:, 1] += w * (preds == 1)
                train_acc = np.mean((weighted_votes[:, 1] > weighted_votes[:, 0]).astype(int) == y)
                print(f"  训练了 {i + 1} 个分类器, 当前准确率: {train_acc:.4f}")
        
        self.is_fitted = True
        print(f"\n训练完成!")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        n_samples = len(X)
        weighted_votes = np.zeros((n_samples, 2))
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions = estimator.predict(X)
            weighted_votes[:, 0] += weight * (predictions == 0)
            weighted_votes[:, 1] += weight * (predictions == 1)
        
        return (weighted_votes[:, 1] > weighted_votes[:, 0]).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
    
    def staged_predict(self, X: np.ndarray) -> List[np.ndarray]:
        """分阶段预测"""
        n_samples = len(X)
        predictions = np.zeros((self.n_estimators, n_samples))
        
        for i, (estimator, weight) in enumerate(zip(self.estimators, self.estimator_weights)):
            preds = estimator.predict(X)
            votes = weight * (2 * preds - 1)
            if i == 0:
                cumulative_votes = votes
            else:
                cumulative_votes += votes
            predictions[i] = (cumulative_votes > 0).astype(int)
        
        return [predictions[i] for i in range(self.n_estimators)]


if __name__ == "__main__":
    print("="*60)
    print("AdaBoost 示例")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 300
    
    # 生成非线性可分数据
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1.5).astype(int)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\n训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 训练 AdaBoost
    ada = AdaBoost(n_estimators=50, random_state=42)
    ada.fit(X_train, y_train)
    
    print(f"\n训练集准确率: {ada.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {ada.score(X_test, y_test):.4f}")
    
    # 分阶段准确率
    print("\n分阶段测试集准确率:")
    for i, (pred, weight, error) in enumerate(zip(
        ada.staged_predict(X_test), ada.estimator_weights, ada.estimator_errors
    )):
        if (i + 1) % 10 == 0:
            acc = np.mean(pred == y_test)
            print(f"  第 {i+1} 个分类器: 准确率={acc:.4f}, 权重={weight:.4f}, 误差={error:.4f}")
    
    print("\n" + "="*60)
