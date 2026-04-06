"""
================================================================================
                        随机森林 - NumPy 实现
================================================================================
集成多棵决策树进行投票/平均预测
================================================================================
"""

import numpy as np
from typing import Optional, List
from .decision_tree import DecisionTree
from .training_framework import Model


class RandomForest(Model):
    """
    随机森林分类器
    
    特点:
        - Bootstrap 有放回采样
        - 随机特征选择
        - 多数投票决策
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', n_classes: int = 2,
                 random_state: Optional[int] = None, n_jobs: int = 1,
                 bootstrap: bool = True):
        """
        参数:
            n_estimators: 树的数量
            max_depth: 每棵树的最大深度
            min_samples_split: 分裂所需最小样本数
            min_samples_leaf: 叶子节点最小样本数
            max_features: 每次分裂考虑的最大特征数
            n_classes: 类别数
            random_state: 随机种子
            n_jobs: 并行工作数 (目前仅支持1)
            bootstrap: 是否使用 bootstrap 采样
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_classes = n_classes
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        
        self.trees: List[DecisionTree] = []
        self.n_features: int = 0
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'RandomForest':
        """
        训练随机森林
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 目标标签 (n_samples,)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        if self.max_features == 'sqrt':
            self.max_features_per_split = max(1, int(np.sqrt(self.n_features)))
        elif self.max_features == 'log2':
            self.max_features_per_split = max(1, int(np.log2(self.n_features)))
        else:
            self.max_features_per_split = self.n_features
        
        print(f"\n{'='*60}")
        print("开始训练随机森林")
        print(f"{'='*60}")
        print(f"样本数: {self.n_samples}, 特征数: {self.n_features}")
        print(f"树的数量: {self.n_estimators}")
        print(f"最大深度: {self.max_depth}")
        print(f"每棵树使用的特征数: {self.max_features_per_split}")
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap 采样
            if self.bootstrap:
                indices = np.random.choice(self.n_samples, self.n_samples, replace=True)
            else:
                indices = np.arange(self.n_samples)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # 训练单棵树
            tree = DecisionTree(
                criterion='gini',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features_per_split,
                random_state=self.random_state + i if self.random_state else None,
            )
            tree.verbose = False
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            if (i + 1) % 20 == 0 or i == 0:
                print(f"已训练 {i + 1}/{self.n_estimators} 棵树...")
        
        self.is_fitted = True
        print(f"\n随机森林训练完成! 共 {len(self.trees)} 棵树")
        
        return self
    
    def _predict_single(self, x: np.ndarray) -> int:
        """单样本预测 - 投票"""
        votes = np.zeros(self.n_classes)
        
        for tree in self.trees:
            prediction = tree._predict_single(x, tree.root)
            votes[prediction] += 1
        
        return np.argmax(votes)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        参数:
            X: 输入数据 (n_samples, n_features)
            
        返回:
            预测类别 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        return np.array([self._predict_single(x) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        返回:
            各类别概率 (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        proba = np.zeros((len(X), self.n_classes))
        
        for tree in self.trees:
            votes = tree.predict(X)
            for i, v in enumerate(votes):
                proba[i, v] += 1
        
        return proba / self.n_estimators
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return np.mean(self.predict(X) == y)
    
    def get_feature_importances(self) -> np.ndarray:
        """获取特征重要性 (所有树的平均)"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        importances = np.zeros(self.n_features)
        for tree in self.trees:
            importances += tree.feature_importances_
        
        return importances / self.n_estimators
    
    def __str__(self) -> str:
        return f"""Random Forest Classifier
================================
Number of Trees: {self.n_estimators}
Max Depth: {self.max_depth}
Number of Classes: {self.n_classes}
"""


# =============================================================================
#                           示例: 使用随机森林
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("随机森林示例 - NumPy 实现")
    print("="*60)
    
    # ========================================
    # 示例 1: 鸢尾花分类
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: 鸢尾花数据集分类")
    print("-"*40)
    
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
        
        # 训练随机森林
        rf = RandomForest(n_estimators=50, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        
        # 评估
        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        
        print(f"\n训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 特征重要性
        print(f"\n特征重要性:")
        importances = rf.get_feature_importances()
        for i, importance in enumerate(importances):
            print(f"  {iris.feature_names[i]}: {importance:.4f}")
        
    except ImportError:
        print("sklearn 不可用，使用模拟数据...")
    
    # ========================================
    # 示例 2: 树数量对性能的影响
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: 树数量对性能的影响")
    print("-"*40)
    
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 10)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0) & (X[:, 2] > 0)).astype(int)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"{'树数量':>10} | {'训练准确率':>12} | {'测试准确率':>12}")
    print("-" * 40)
    
    for n_trees in [5, 10, 20, 50, 100]:
        rf = RandomForest(n_estimators=n_trees, max_depth=10, random_state=42)
        rf.fit(X_train, y_train, X_test, y_test)
        
        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        
        print(f"{n_trees:>10} | {train_acc:>12.4f} | {test_acc:>12.4f}")
    
    # ========================================
    # 示例 3: OOB 误差估计
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: 袋外 (OOB) 误差估计")
    print("-"*40)
    
    # 简化的 OOB 估计
    n_samples_oob = 100
    X_oob = np.random.randn(n_samples_oob, 5)
    y_oob = ((X_oob[:, 0] > 0) | (X_oob[:, 1] > 0)).astype(int)
    
    rf_oob = RandomForest(n_estimators=30, max_depth=8, bootstrap=True, random_state=42)
    rf_oob.fit(X[:split], y[:split])
    
    # 计算 OOB 预测
    oob_predictions = np.zeros(n_samples_oob)
    oob_counts = np.zeros(n_samples_oob)
    
    # 对每个样本，统计没有使用它的树做出的预测
    np.random.seed(42)
    for i in range(rf_oob.n_estimators):
        indices = np.random.choice(rf_oob.n_samples, rf_oob.n_samples, replace=True)
        oob_mask = np.setdiff1d(np.arange(rf_oob.n_samples), np.unique(indices))
        
        for idx in oob_mask:
            if idx < len(X_oob):
                oob_predictions[idx] += rf_oob.trees[i]._predict_single(X_oob[idx], rf_oob.trees[i].root)
                oob_counts[idx] += 1
    
    # 计算 OOB 准确率
    valid_mask = oob_counts > 0
    if np.sum(valid_mask) > 0:
        oob_preds = (oob_predictions[valid_mask] / oob_counts[valid_mask] > 0.5).astype(int)
        oob_acc = np.mean(oob_preds == y_oob[valid_mask])
        print(f"袋外 (OOB) 准确率: {oob_acc:.4f}")
        print(f"被估计的样本数: {np.sum(valid_mask)}/{n_samples_oob}")
    
    print("\n" + "="*60)
    print("随机森林示例完成!")
    print("="*60)
