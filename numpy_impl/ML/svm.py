"""
================================================================================
                        支持向量机 (SVM) - NumPy 实现
================================================================================
包含:
    - 硬间隔 SVM
    - 软间隔 SVM
    - 核函数 (线性、多项式、高斯 RBF)
================================================================================
"""

import numpy as np
from typing import Optional, Tuple, Callable
from .training_framework import Model


class SVM(Model):
    """
    支持向量机分类器
    
    支持:
        - 硬间隔和软间隔
        - 线性核、多项式核、高斯核
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear',
                 degree: int = 3, gamma: float = 1.0, coef0: float = 1.0,
                 max_iter: int = 1000, tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """
        参数:
            C: 正则化参数 (软间隔)
            kernel: 核函数类型 ('linear', 'poly', 'rbf')
            degree: 多项式核的度数
            gamma: RBF 核参数
            coef0: 核函数常数项
            max_iter: SMO 算法最大迭代次数
            tol: 收敛阈值
            random_state: 随机种子
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.alphas: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.support_vectors_idx: Optional[np.ndarray] = None
        self.n_support: int = 0
        self.is_fitted = False
    
    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """核函数"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            diff = x1 - x2
            return np.exp(-self.gamma * np.dot(diff, diff))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算核矩阵"""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel(X1[i], X2[j])
        
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
            ) -> 'SVM':
        """
        使用 SMO 算法训练 SVM
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 目标标签 (-1 或 +1)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_samples, self.n_features = X.shape
        
        # 转换标签为 -1 和 +1
        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            if unique_labels[0] == 0 and unique_labels[1] == 1:
                y = 2 * y - 1
        
        self.X_train = X
        self.y_train = y
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("开始训练 SVM")
            print(f"{'='*60}")
            print(f"样本数: {self.n_samples}, 特征数: {self.n_features}")
            print(f"C: {self.C}, Kernel: {self.kernel}")
        
        # 初始化
        self.alphas = np.zeros(self.n_samples)
        self.b = 0.0
        
        # 预计算核矩阵
        if self.kernel != 'linear':
            self.K = self._compute_kernel_matrix(X, X)
        else:
            self.K = np.dot(X, X.T)
        
        # SMO 算法
        passes = 0
        max_passes = 5
        
        while passes < max_passes:
            num_changed_alphas = 0
            
            for i in range(self.n_samples):
                # 计算 Ei
                f_xi = self._predict_single_alpha(X[i])
                Ei = f_xi - y[i]
                
                # KKT 条件检查
                if (y[i] * Ei < -self.tol and self.alphas[i] < self.C) or \
                   (y[i] * Ei > self.tol and self.alphas[i] > 0):
                    
                    # 选择 j != i
                    j = np.random.choice([k for k in range(self.n_samples) if k != i])
                    
                    # 计算 Ej
                    f_xj = self._predict_single_alpha(X[j])
                    Ej = f_xj - y[j]
                    
                    # 保存旧值
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # 计算边界
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # 计算 eta
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue
                    
                    # 更新 alpha_j
                    self.alphas[j] = alpha_j_old - (y[j] * (Ei - Ej)) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 更新 alpha_i
                    self.alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # 更新 b
                    b1 = self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, i] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, j] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * self.K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        # 找到支持向量
        self.support_vectors_idx = np.where(self.alphas > 1e-7)[0]
        self.n_support = len(self.support_vectors_idx)
        
        self.is_fitted = True
        
        if self.verbose:
            print(f"训练完成!")
            print(f"支持向量数量: {self.n_support}")
            print(f"支持向量比例: {self.n_support / self.n_samples:.2%}")
        
        return self
    
    verbose = True
    
    def _predict_single_alpha(self, x: np.ndarray) -> float:
        """使用 alpha 值预测单个样本"""
        result = 0.0
        for i in range(self.n_samples):
            result += self.alphas[i] * self.y_train[i] * self._kernel(x, self.X_train[i])
        return result + self.b
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        参数:
            X: 输入数据 (n_samples, n_features)
            
        返回:
            预测类别 (n_samples,), 标签为 0 或 1
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            f = self._predict_single_alpha(x)
            predictions[i] = np.sign(f)
        
        # 转换回 0/1 标签
        predictions = (predictions + 1) / 2
        return predictions.astype(int)
    
    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """预测原始分数"""
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            predictions[i] = self._predict_single_alpha(x)
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return np.mean(self.predict(X) == y)
    
    def get_support_vectors(self) -> np.ndarray:
        """获取支持向量"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.X_train[self.support_vectors_idx]
    
    def __str__(self) -> str:
        return f"""SVM Classifier
======================
C: {self.C}
Kernel: {self.kernel}
Support Vectors: {self.n_support}
"""


# =============================================================================
#                           示例: 使用 SVM
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("SVM 示例 - NumPy 实现")
    print("="*60)
    
    # ========================================
    # 示例 1: 线性可分数据
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: 线性可分数据")
    print("-"*40)
    
    np.random.seed(42)
    n_samples = 200
    
    # 生成线性可分数据
    X1 = np.random.randn(n_samples//2, 2) + [2, 2]
    X2 = np.random.randn(n_samples//2, 2) + [-2, -2]
    X = np.vstack([X1, X2])
    y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
    
    # 打乱
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 训练 SVM
    svm = SVM(C=1.0, kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    
    print(f"\n训练集准确率: {svm.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {svm.score(X_test, y_test):.4f}")
    
    # ========================================
    # 示例 2: 核函数对比
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: 不同核函数对比")
    print("-"*40)
    
    # 生成非线性可分数据 (圆形)
    np.random.seed(42)
    n_samples = 300
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    r = 5 + np.random.randn(n_samples) * 0.5
    
    X_circle = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # 内圈和外圈
    mask_inner = r < 5
    mask_outer = r >= 5
    y_circle = mask_inner.astype(int)
    
    # 添加噪声
    X_circle += np.random.randn(n_samples, 2) * 0.3
    
    split = int(0.8 * n_samples)
    X_train_c, X_test_c = X_circle[:split], X_circle[split:]
    y_train_c, y_test_c = y_circle[:split], y_circle[split:]
    
    print(f"\n数据点: {n_samples}, 内圈: {np.sum(mask_inner)}, 外圈: {np.sum(mask_outer)}")
    
    # 线性核
    svm_linear = SVM(C=1.0, kernel='linear', random_state=42)
    svm_linear.fit(X_train_c, y_train_c)
    
    # RBF 核
    svm_rbf = SVM(C=1.0, kernel='rbf', gamma=0.5, random_state=42)
    svm_rbf.fit(X_train_c, y_train_c)
    
    # 多项式核
    svm_poly = SVM(C=1.0, kernel='poly', degree=3, random_state=42)
    svm_poly.fit(X_train_c, y_train_c)
    
    print(f"\n线性核 - 训练: {svm_linear.score(X_train_c, y_train_c):.4f}, "
          f"测试: {svm_linear.score(X_test_c, y_test_c):.4f}")
    print(f"RBF核 - 训练: {svm_rbf.score(X_train_c, y_train_c):.4f}, "
          f"测试: {svm_rbf.score(X_test_c, y_test_c):.4f}")
    print(f"多项式核 - 训练: {svm_poly.score(X_train_c, y_train_c):.4f}, "
          f"测试: {svm_poly.score(X_test_c, y_test_c):.4f}")
    
    # ========================================
    # 示例 3: C 参数影响
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: C 参数对软间隔 SVM 的影响")
    print("-"*40)
    
    print(f"\n{'C 值':>10} | {'训练准确率':>12} | {'测试准确率':>12} | {'支持向量数':>12}")
    print("-" * 52)
    
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        svm_c = SVM(C=C, kernel='rbf', gamma=0.5, random_state=42)
        svm_c.fit(X_train_c, y_train_c)
        
        train_acc = svm_c.score(X_train_c, y_train_c)
        test_acc = svm_c.score(X_test_c, y_test_c)
        n_sv = svm_c.n_support
        
        print(f"{C:>10.2f} | {train_acc:>12.4f} | {test_acc:>12.4f} | {n_sv:>12d}")
    
    print("\n" + "="*60)
    print("SVM 示例完成!")
    print("="*60)
