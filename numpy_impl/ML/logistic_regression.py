"""
================================================================================
                        逻辑回归 - NumPy 实现
================================================================================
包含:
    - 二分类逻辑回归
    - 多分类逻辑回归 (One-vs-All, Softmax)
    - 梯度下降训练
    - 正则化 (L1, L2)
================================================================================
"""

import numpy as np
from typing import Optional, Tuple, List
from .training_framework import Model
from .losses import CrossEntropyLoss, BinaryCrossEntropyLoss


class LogisticRegression(Model):
    """
    逻辑回归分类器
    
    支持:
        - 二分类和多分类
        - 梯度下降优化
        - L1/L2 正则化
        - 早停机制
    """
    
    def __init__(self, input_dim: int, num_classes: int = 2,
                 regularization: str = 'l2', alpha: float = 0.01,
                 learning_rate: float = 0.1, max_iter: int = 1000,
                 tol: float = 1e-6, early_stopping: bool = True,
                 verbose: bool = True):
        """
        参数:
            input_dim: 输入特征维度
            num_classes: 类别数量 (2 表示二分类)
            regularization: 正则化类型 ('l1', 'l2', 'none')
            alpha: 正则化强度
            learning_rate: 学习率
            max_iter: 最大迭代次数
            tol: 收敛阈值
            early_stopping: 是否使用早停
            verbose: 是否打印训练过程
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.regularization = regularization
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.verbose = verbose
        
        # 初始化权重 (Xavier 初始化)
        scale = np.sqrt(2.0 / (input_dim + num_classes))
        if num_classes == 2:
            self.weights = np.random.randn(input_dim) * scale
            self.bias = 0.0
        else:
            self.weights = np.random.randn(input_dim, num_classes) * scale
            self.bias = np.zeros(num_classes)
        
        # 训练历史
        self.loss_history = []
        self.accuracy_history = []
        self.is_fitted = False
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid 函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.num_classes == 2:
            z = np.dot(X, self.weights) + self.bias
            proba = self._sigmoid(z)
            return np.column_stack([1 - proba, proba])
        else:
            z = np.dot(X, self.weights) + self.bias
            return self._softmax(z)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播 (返回类别预测)"""
        if self.num_classes == 2:
            proba = self._predict_proba(X)[:, 1]
            return (proba >= 0.5).astype(int)
        else:
            proba = self._predict_proba(X)
            return np.argmax(proba, axis=1)
    
    def backward(self, grad_output: np.ndarray, learning_rate: float):
        """反向传播 (供 Trainer 使用)"""
        pass  # 逻辑回归在 fit 中直接进行梯度下降
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'LogisticRegression':
        """
        训练模型
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 目标标签 (n_samples,), 0 到 num_classes-1
            X_val: 验证数据 (可选)
            y_val: 验证标签 (可选)
        """
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("开始训练逻辑回归")
            print(f"{'='*60}")
            print(f"样本数: {n_samples}, 特征数: {n_features}")
            print(f"类别数: {self.num_classes}")
            print(f"学习率: {self.learning_rate}, 最大迭代: {self.max_iter}")
            print(f"正则化: {self.regularization}, alpha: {self.alpha}")
        
        best_val_acc = 0.0
        patience = 0
        max_patience = 50
        
        for iteration in range(self.max_iter):
            # 前向传播
            if self.num_classes == 2:
                z = np.dot(X, self.weights) + self.bias
                proba = self._sigmoid(z)
                
                # 计算损失
                eps = 1e-15
                proba = np.clip(proba, eps, 1 - eps)
                loss = -np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba))
                
                # 梯度计算
                grad = (proba - y) / n_samples
                grad_weights = np.dot(X.T, grad)
                grad_bias = np.mean(grad)
            else:
                z = np.dot(X, self.weights) + self.bias
                proba = self._softmax(z)
                
                # 计算损失 (交叉熵)
                eps = 1e-15
                proba = np.clip(proba, eps, 1 - eps)
                loss = -np.sum(np.log(proba[np.arange(n_samples), y])) / n_samples
                
                # 梯度计算
                grad = proba.copy()
                grad[np.arange(n_samples), y] -= 1
                grad_weights = np.dot(X.T, grad) / n_samples
                grad_bias = np.mean(grad, axis=0)
            
            # 添加正则化
            if self.regularization == 'l2':
                if self.num_classes == 2:
                    grad_weights += self.alpha * self.weights
                else:
                    grad_weights += self.alpha * self.weights
                loss += (self.alpha / 2) * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                if self.num_classes == 2:
                    grad_weights += self.alpha * np.sign(self.weights)
                else:
                    grad_weights += self.alpha * np.sign(self.weights)
                loss += self.alpha * np.sum(np.abs(self.weights))
            
            # 更新权重
            if self.num_classes == 2:
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
            else:
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
            
            # 记录历史
            self.loss_history.append(loss)
            train_acc = self._accuracy(X, y)
            self.accuracy_history.append(train_acc)
            
            # 早停检查
            if self.early_stopping and X_val is not None:
                val_acc = self._accuracy(X_val, y_val)
                if val_acc < best_val_acc:
                    patience += 1
                    if patience >= max_patience:
                        if self.verbose:
                            print(f"\n早停于第 {iteration + 1} 次迭代")
                        break
                else:
                    best_val_acc = val_acc
                    patience = 0
            
            # 打印进度
            if self.verbose and (iteration + 1) % max(1, self.max_iter // 10) == 0:
                val_info = f", Val Acc: {self._accuracy(X_val, y_val):.4f}" if X_val is not None else ""
                print(f"迭代 {iteration + 1:5d}/{self.max_iter} | "
                      f"Loss: {loss:.4f} | Train Acc: {train_acc:.4f}{val_info}")
            
            # 收敛检查
            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                if self.verbose:
                    print(f"\n在第 {iteration + 1} 次迭代收敛")
                break
        
        self.is_fitted = True
        return self
    
    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        predictions = self.forward(X)
        return np.mean(predictions == y)
    
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
        return self.forward(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        参数:
            X: 输入数据 (n_samples, n_features)
            
        返回:
            各类别概率 (n_samples, num_classes)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        return self._predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return self._accuracy(X, y)
    
    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取模型参数"""
        if self.num_classes == 2:
            return self.weights.copy(), np.array([self.bias])
        return self.weights.copy(), self.bias.copy()
    
    def __str__(self) -> str:
        return f"""Logistic Regression Model
================================
Input Dimension: {self.input_dim}
Number of Classes: {self.num_classes}
Regularization: {self.regularization}
Alpha: {self.alpha}
"""


# =============================================================================
#                           示例: 使用逻辑回归
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("逻辑回归示例 - NumPy 实现")
    print("="*60)
    
    # ========================================
    # 示例 1: 二分类 - 鸢尾花数据集
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: 二分类 (鸢尾花数据)")
    print("-"*40)
    
    # 使用 sklearn 的数据作为参考
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # 加载数据
        iris = load_iris()
        X = iris.data[iris.target != 2]  # 只取两类
        y = iris.target[iris.target != 2]
        y = (y == 2).astype(int)  # 转换为 0/1
        
        feature_names = iris.feature_names
        target_names = iris.target_names[:2]
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        print(f"类别分布: {np.bincount(y_train)}")
        
        # 训练模型
        model = LogisticRegression(input_dim=X_train.shape[1], num_classes=2,
                                   learning_rate=0.1, max_iter=1000)
        model.fit(X_train, y_train, X_test, y_test)
        
        # 评估
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"\n训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 预测示例
        proba = model.predict_proba(X_test[:5])
        pred = model.predict(X_test[:5])
        print(f"\n测试样本预测 (前5个):")
        print(f"概率: {proba}")
        print(f"预测: {pred}")
        
    except ImportError:
        print("sklearn 不可用，使用模拟数据...")
        
        # 模拟数据
        np.random.seed(42)
        n_samples = 300
        
        # 两类数据
        X1 = np.random.randn(n_samples//2, 2) + [2, 2]
        X2 = np.random.randn(n_samples//2, 2) + [-2, -2]
        X = np.vstack([X1, X2])
        y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
        
        # 打乱
        idx = np.random.permutation(n_samples)
        X, y = X[idx], y[idx]
        
        # 划分
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # 训练
        model = LogisticRegression(input_dim=2, num_classes=2,
                                   learning_rate=0.1, max_iter=500)
        model.fit(X_train, y_train, X_test, y_test)
        
        print(f"训练集准确率: {model.score(X_train, y_train):.4f}")
        print(f"测试集准确率: {model.score(X_test, y_test):.4f}")
    
    # ========================================
    # 示例 2: 多分类 - 手写数字
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: 多分类 (手写数字识别)")
    print("-"*40)
    
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        
        # 加载数据
        digits = load_digits()
        X = digits.data / 16.0  # 归一化
        y = digits.target
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        print(f"类别数: {len(np.unique(y))}")
        
        # 训练模型
        model_multi = LogisticRegression(input_dim=X_train.shape[1], num_classes=10,
                                        learning_rate=0.1, max_iter=500)
        model_multi.fit(X_train, y_train, X_test, y_test)
        
        # 评估
        train_acc = model_multi.score(X_train, y_train)
        test_acc = model_multi.score(X_test, y_test)
        print(f"\n训练集准确率: {train_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        
    except ImportError:
        print("sklearn 不可用，跳过手写数字识别示例")
    
    # ========================================
    # 示例 3: 正则化对比
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: L1 vs L2 正则化对比")
    print("-"*40)
    
    np.random.seed(42)
    n_samples = 200
    n_features = 20  # 高维数据
    
    # 生成相关特征数据
    X_reg = np.random.randn(n_samples, n_features)
    X_reg[:, :5] = X_reg[:, 0:1] + np.random.randn(n_samples, 5) * 0.1
    y_reg = (X_reg[:, 0] + X_reg[:, 1] > 0).astype(int)
    
    split = int(0.8 * n_samples)
    X_train_r, X_test_r = X_reg[:split], X_reg[split:]
    y_train_r, y_test_r = y_reg[:split], y_reg[split:]
    
    # L2 正则化
    model_l2 = LogisticRegression(input_dim=n_features, num_classes=2,
                                   regularization='l2', alpha=0.1)
    model_l2.fit(X_train_r, y_train_r)
    
    # L1 正则化
    model_l1 = LogisticRegression(input_dim=n_features, num_classes=2,
                                   regularization='l1', alpha=0.1)
    model_l1.fit(X_train_r, y_train_r)
    
    print(f"L2 正则化 - 训练准确率: {model_l2.score(X_train_r, y_train_r):.4f}, "
          f"测试准确率: {model_l2.score(X_test_r, y_test_r):.4f}")
    print(f"L2 非零权重数: {np.sum(np.abs(model_l2.get_weights()[0]) > 0.01)}/{n_features}")
    
    print(f"L1 正则化 - 训练准确率: {model_l1.score(X_train_r, y_train_r):.4f}, "
          f"测试准确率: {model_l1.score(X_test_r, y_test_r):.4f}")
    print(f"L1 非零权重数: {np.sum(np.abs(model_l1.get_weights()[0]) > 0.01)}/{n_features}")
    
    print("\n" + "="*60)
    print("逻辑回归示例完成!")
    print("="*60)
