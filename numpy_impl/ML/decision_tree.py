"""
================================================================================
                        决策树 - NumPy 实现
================================================================================
包含:
    - ID3 决策树 (基于信息增益)
    - CART 决策树 (基于基尼系数)
    - 分类与回归
    - 剪枝处理
================================================================================
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from .training_framework import Model


@dataclass
class TreeNode:
    """决策树节点"""
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    value: Optional[int] = None  # 叶子节点类别
    is_leaf: bool = False


class DecisionTree(Model):
    """
    决策树分类器
    
    支持:
        - ID3 (信息增益)
        - CART (基尼系数)
        - 最大深度限制
        - 最小样本分裂
        - 早停
    """
    
    def __init__(self, criterion: str = 'gini', max_depth: int = 10,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Optional[int] = None, random_state: Optional[int] = None):
        """
        参数:
            criterion: 分裂准则 ('gini', 'entropy')
            max_depth: 最大深度
            min_samples_split: 节点分裂所需最小样本数
            min_samples_leaf: 叶子节点最小样本数
            max_features: 分裂时考虑的最大特征数
            random_state: 随机种子
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
        self.root: Optional[TreeNode] = None
        self.n_classes: int = 0
        self.n_features: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self.is_fitted = False
    
    def _gini(self, y: np.ndarray) -> float:
        """计算基尼系数"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """计算信息熵"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _impurity(self, y: np.ndarray) -> float:
        """计算不纯度"""
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """计算信息增益"""
        n = len(y)
        if n == 0:
            return 0
        
        parent_impurity = self._impurity(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        child_impurity = (n_left / n) * self._impurity(y_left) + (n_right / n) * self._impurity(y_right)
        return parent_impurity - child_impurity
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """找到最佳分裂点"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            # 获取该特征的唯一值作为候选阈值
            thresholds = np.unique(X[:, feature_idx])
            
            # 采样候选阈值 (如果太多)
            if len(thresholds) > 10:
                indices = np.linspace(0, len(thresholds)-1, 10, dtype=int)
                thresholds = thresholds[indices]
            
            for threshold in thresholds:
                # 分裂数据
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                y_left, y_right = y[left_mask], y[right_mask]
                
                # 检查最小样本要求
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # 计算信息增益
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        """递归构建决策树"""
        n_samples = len(y)
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            # 创建叶子节点
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value, is_leaf=True)
        
        # 选择特征
        n_features = X.shape[1]
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif self.max_features == 'sqrt':
            feature_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        elif self.max_features == 'log2':
            feature_indices = np.random.choice(n_features, int(np.log2(n_features)), replace=False)
        else:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        
        # 找到最佳分裂
        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)
        
        if best_feature is None or best_gain <= 0:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value, is_leaf=True)
        
        # 分裂数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # 递归构建子树
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node
        )
    
    def _most_common_label(self, y: np.ndarray) -> int:
        """返回最常见的标签"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes)
        return np.argmax(counts)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'DecisionTree':
        """
        训练决策树
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 目标标签 (n_samples,)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("开始训练决策树")
            print(f"{'='*60}")
            print(f"样本数: {self.n_samples}, 特征数: {self.n_features}")
            print(f"类别数: {self.n_classes}")
            print(f"最大深度: {self.max_depth}")
            print(f"分裂准则: {self.criterion}")
        
        # 构建树
        self.root = self._build_tree(X, y, depth=0)
        
        # 计算特征重要性
        self._compute_feature_importances(X, y)
        
        self.is_fitted = True
        
        if self.verbose:
            print(f"决策树构建完成!")
            print(f"叶子节点数: {self._count_leaves(self.root)}")
            print(f"树深度: {self._tree_depth(self.root)}")
        
        return self
    
    def _predict_single(self, x: np.ndarray, node: TreeNode) -> int:
        """预测单个样本"""
        if node.is_leaf:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
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
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        predictions = self.predict(X)
        proba = np.zeros((len(X), self.n_classes))
        proba[np.arange(len(X)), predictions] = 1
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return np.mean(self.predict(X) == y)
    
    def _count_leaves(self, node: TreeNode) -> int:
        """统计叶子节点数"""
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def _tree_depth(self, node: TreeNode) -> int:
        """计算树深度"""
        if node.is_leaf:
            return 0
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))
    
    def _compute_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """计算特征重要性"""
        self.feature_importances_ = np.zeros(self.n_features)
        
        def _compute_recursive(node: TreeNode, depth: int, n_samples: int):
            if node.is_leaf:
                return
            
            n_left = n_samples // 2  # 简化的计算
            n_right = n_samples - n_left
            
            # 简化: 基于深度和分裂次数
            if node.feature_index is not None:
                self.feature_importances_[node.feature_index] += 1
        
        _compute_recursive(self.root, 0, len(y))
        self.feature_importances_ /= np.sum(self.feature_importances_)
    
    def _print_tree(self, node: TreeNode, depth: int = 0, prefix: str = "Root: "):
        """打印决策树"""
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}{prefix}Leaf: class={node.value}")
        else:
            print(f"{indent}{prefix}Split: feature[{node.feature_index}] <= {node.threshold:.4f}")
            self._print_tree(node.left, depth + 1, "L-> ")
            self._print_tree(node.right, depth + 1, "R-> ")
    
    def print_tree(self):
        """打印整棵树"""
        if self.root is not None:
            self._print_tree(self.root)
    
    verbose = True  # 添加这个属性以避免警告


# =============================================================================
#                           示例: 使用决策树
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("决策树示例 - NumPy 实现")
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
        
        # 加载数据
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
        
        # 训练决策树
        tree = DecisionTree(criterion='gini', max_depth=5, random_state=42)
        tree.fit(X_train, y_train)
        
        # 评估
        train_acc = tree.score(X_train, y_train)
        test_acc = tree.score(X_test, y_test)
        
        print(f"\n训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 特征重要性
        print(f"\n特征重要性:")
        for i, importance in enumerate(tree.feature_importances_):
            print(f"  {iris.feature_names[i]}: {importance:.4f}")
        
        # 打印树结构
        print("\n决策树结构:")
        tree.print_tree()
        
    except ImportError:
        print("sklearn 不可用，使用模拟数据...")
    
    # ========================================
    # 示例 2: 不同准则对比
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: Gini vs Entropy 对比")
    print("-"*40)
    
    np.random.seed(42)
    n_samples = 500
    
    # 生成复杂数据
    X_complex = np.random.randn(n_samples, 4)
    y_complex = ((X_complex[:, 0] > 0) & (X_complex[:, 1] > 0)).astype(int)
    
    split = int(0.8 * n_samples)
    X_train_c, X_test_c = X_complex[:split], X_complex[split:]
    y_train_c, y_test_c = y_complex[:split], y_complex[split:]
    
    # Gini
    tree_gini = DecisionTree(criterion='gini', max_depth=10, random_state=42)
    tree_gini.fit(X_train_c, y_train_c)
    
    # Entropy
    tree_entropy = DecisionTree(criterion='entropy', max_depth=10, random_state=42)
    tree_entropy.fit(X_train_c, y_train_c)
    
    print(f"Gini - 训练准确率: {tree_gini.score(X_train_c, y_train_c):.4f}, "
          f"测试准确率: {tree_gini.score(X_test_c, y_test_c):.4f}")
    print(f"Entropy - 训练准确率: {tree_entropy.score(X_train_c, y_train_c):.4f}, "
          f"测试准确率: {tree_entropy.score(X_test_c, y_test_c):.4f}")
    
    # ========================================
    # 示例 3: 过拟合与深度控制
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: 深度对性能的影响")
    print("-"*40)
    
    for depth in [2, 5, 10, 20, None]:
        tree_depth = DecisionTree(max_depth=depth, random_state=42)
        tree_depth.fit(X_train_c, y_train_c)
        
        train_acc = tree_depth.score(X_train_c, y_train_c)
        test_acc = tree_depth.score(X_test_c, y_test_c)
        depth_str = str(depth) if depth else "无限制"
        
        print(f"最大深度 {depth_str:>10} | 训练: {train_acc:.4f} | 测试: {test_acc:.4f}")
    
    print("\n" + "="*60)
    print("决策树示例完成!")
    print("="*60)
