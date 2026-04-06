"""
================================================================================
                        线性回归 - NumPy 实现
================================================================================
包含:
    - 简单线性回归
    - 多元线性回归
    - 岭回归 (Ridge Regression)
    - LASSO 回归
    - 梯度下降法 / 正规方程求解
================================================================================
"""

import numpy as np
from typing import Optional, Tuple
from .training_framework import Model


class LinearRegression(Model):
    """
    线性回归模型
    
    支持:
        - 梯度下降训练
        - 正规方程求解
        - 岭回归 (L2 正则化)
        - LASSO (L1 正则化)
    """
    
    def __init__(self, input_dim: int = 1, regularization: str = 'none', 
                 alpha: float = 0.01, learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-6,
                 method: str = 'gd', early_stopping: bool = False):
        """
        参数:
            input_dim: 输入特征维度
            regularization: 正则化类型 ('none', 'ridge', 'lasso')
            alpha: 正则化强度
            learning_rate: 学习率 (梯度下降时使用)
            max_iter: 最大迭代次数
            tol: 收敛阈值
            method: 求解方法 ('gd': 梯度下降, 'normal': 正规方程)
            early_stopping: 是否使用早停
        """
        self.input_dim = input_dim
        self.regularization = regularization
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.early_stopping = early_stopping
        
        # 初始化参数
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0
        
        # 训练历史
        self.loss_history = []
        self.is_fitted = False
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """添加偏置项"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算 MSE 损失"""
        predictions = self.predict_raw(X)
        mse = np.mean((predictions - y) ** 2)
        
        # 添加正则化项
        if self.regularization == 'ridge':
            mse += self.alpha * np.sum(self.weights ** 2)
        elif self.regularization == 'lasso':
            mse += self.alpha * np.sum(np.abs(self.weights))
        
        return mse
    
    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """原始预测"""
        return np.dot(X, self.weights) + self.bias
    
    def fit_gd(self, X: np.ndarray, y: np.ndarray, 
               X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
               verbose: bool = True) -> 'LinearRegression':
        """
        使用梯度下降法训练
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 目标值 (n_samples,)
            X_val: 验证数据 (可选)
            y_val: 验证目标值 (可选)
            verbose: 是否打印训练过程
        """
        n_samples, n_features = X.shape
        
        if verbose:
            print(f"\n{'='*60}")
            print("开始训练线性回归 (梯度下降法)")
            print(f"{'='*60}")
            print(f"样本数: {n_samples}, 特征数: {n_features}")
            print(f"学习率: {self.learning_rate}, 最大迭代: {self.max_iter}")
            print(f"正则化: {self.regularization}, alpha: {self.alpha}")
        
        prev_loss = float('inf')
        patience = 0
        max_patience = 50
        
        for iteration in range(self.max_iter):
            # 前向传播
            predictions = self.predict_raw(X)
            
            # 计算梯度
            error = predictions - y
            grad_weights = (2 / n_samples) * np.dot(X.T, error)
            grad_bias = (2 / n_samples) * np.sum(error)
            
            # 添加正则化梯度
            if self.regularization == 'ridge':
                grad_weights += 2 * self.alpha * self.weights
            elif self.regularization == 'lasso':
                grad_weights += self.alpha * np.sign(self.weights)
            
            # 更新参数
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
            
            # 记录损失
            train_loss = self._compute_loss(X, y)
            self.loss_history.append(train_loss)
            
            # 早停检查
            if self.early_stopping and X_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                if val_loss > prev_loss:
                    patience += 1
                    if patience >= max_patience:
                        if verbose:
                            print(f"\n早停于第 {iteration + 1} 次迭代")
                        break
                else:
                    patience = 0
                prev_loss = val_loss
            
            # 收敛检查
            if abs(prev_loss - train_loss) < self.tol:
                if verbose:
                    print(f"\n在第 {iteration + 1} 次迭代收敛")
                break
            
            prev_loss = train_loss
            
            # 打印进度
            if verbose and (iteration + 1) % max(1, self.max_iter // 10) == 0:
                val_info = f", Val Loss: {self._compute_loss(X_val, y_val):.6f}" if X_val is not None else ""
                print(f"迭代 {iteration + 1:5d}/{self.max_iter} | "
                      f"Train Loss: {train_loss:.6f}{val_info}")
        
        self.is_fitted = True
        return self
    
    def fit_normal(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        使用正规方程求解
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 目标值 (n_samples,)
        """
        n_samples, n_features = X.shape
        
        print(f"\n{'='*60}")
        print("开始训练线性回归 (正规方程法)")
        print(f"{'='*60}")
        
        # 添加偏置项
        X_b = self._add_intercept(X)
        
        if self.regularization == 'none':
            # 标准正规方程: theta = (X^T X)^(-1) X^T y
            self.params = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = self.params[0]
            self.weights = self.params[1:]
        elif self.regularization == 'ridge':
            # 岭回归: theta = (X^T X + lambda*I)^(-1) X^T y
            I = np.eye(n_features + 1)
            I[0, 0] = 0  # 不对偏置项正则化
            self.params = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
            self.bias = self.params[0]
            self.weights = self.params[1:]
        
        self.is_fitted = True
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: bool = True) -> 'LinearRegression':
        """
        训练模型
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 目标值 (n_samples,)
            X_val: 验证数据 (可选)
            y_val: 验证目标值 (可选)
            verbose: 是否打印训练过程
        """
        if self.method == 'normal' and self.regularization != 'lasso':
            return self.fit_normal(X, y)
        else:
            return self.fit_gd(X, y, X_val, y_val, verbose)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            X: 输入数据 (n_samples, n_features)
            
        返回:
            预测值 (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用 fit() 方法")
        return self.predict_raw(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算 R² 分数
        
        参数:
            X: 输入数据
            y: 真实目标值
            
        返回:
            R² 分数
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_weights(self) -> Tuple[np.ndarray, float]:
        """获取模型参数"""
        return self.weights.copy(), self.bias
    
    def __str__(self) -> str:
        info = f"""Linear Regression Model
================================
Input Dimension: {self.input_dim}
Method: {self.method}
Regularization: {self.regularization}
Alpha: {self.alpha}
Weights: {self.weights[:5]}... (if more than 5 features)
Bias: {self.bias:.6f}
"""
        return info


# =============================================================================
#                           示例: 使用线性回归
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("线性回归示例 - NumPy 实现")
    print("="*60)
    
    # ========================================
    # 示例 1: 简单线性回归
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: 简单线性回归")
    print("-"*40)
    
    # 生成数据: y = 2x + 3 + noise
    np.random.seed(42)
    n_samples = 200
    X_simple = np.random.randn(n_samples, 1) * 5
    y_simple = 2 * X_simple.flatten() + 3 + np.random.randn(n_samples) * 2
    
    # 划分训练集和测试集
    split = int(0.8 * n_samples)
    X_train, X_test = X_simple[:split], X_simple[split:]
    y_train, y_test = y_simple[:split], y_simple[split:]
    
    # 训练模型
    model = LinearRegression(input_dim=1, method='gd', learning_rate=0.01, max_iter=1000)
    model.fit(X_train, y_train, X_test, y_test)
    
    # 评估
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    weights, bias = model.get_weights()
    
    print(f"\n真实参数: w=2.0, b=3.0")
    print(f"学习参数: w={weights[0]:.4f}, b={bias:.4f}")
    print(f"训练集 R²: {train_score:.4f}")
    print(f"测试集 R²: {test_score:.4f}")
    
    # ========================================
    # 示例 2: 多元线性回归
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: 多元线性回归")
    print("-"*40)
    
    # 生成数据: y = 1 + 2*x1 + 3*x2 + 4*x3 + noise
    np.random.seed(42)
    n_samples = 500
    X_multi = np.random.randn(n_samples, 3) * 10
    y_multi = 1 + 2*X_multi[:, 0] + 3*X_multi[:, 1] + 4*X_multi[:, 2] + np.random.randn(n_samples) * 5
    
    # 划分数据
    split = int(0.8 * n_samples)
    X_train_m, X_test_m = X_multi[:split], X_multi[split:]
    y_train_m, y_test_m = y_multi[:split], y_multi[split:]
    
    # 训练
    model_multi = LinearRegression(input_dim=3, method='normal')
    model_multi.fit(X_train_m, y_train_m)
    
    # 评估
    train_score_m = model_multi.score(X_train_m, y_train_m)
    test_score_m = model_multi.score(X_test_m, y_test_m)
    weights_m, bias_m = model_multi.get_weights()
    
    print(f"\n真实参数: w=[2, 3, 4], b=1")
    print(f"学习参数: w={weights_m}, b={bias_m:.4f}")
    print(f"训练集 R²: {train_score_m:.4f}")
    print(f"测试集 R²: {test_score_m:.4f}")
    
    # ========================================
    # 示例 3: 岭回归对比
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: 岭回归 vs 普通线性回归")
    print("-"*40)
    
    # 生成病态数据 (特征相关性强)
    np.random.seed(42)
    n_samples = 100
    X_ill = np.random.randn(n_samples, 5)
    X_ill[:, 1] = X_ill[:, 0] + np.random.randn(n_samples) * 0.1  # 强相关
    y_ill = 2*X_ill[:, 0] + 3*X_ill[:, 1] + np.random.randn(n_samples) * 0.5
    
    split = int(0.8 * n_samples)
    X_train_r, X_test_r = X_ill[:split], X_ill[split:]
    y_train_r, y_test_r = y_ill[:split], y_ill[split:]
    
    # 普通线性回归
    model_lr = LinearRegression(input_dim=5, method='normal')
    model_lr.fit(X_train_r, y_train_r)
    
    # 岭回归
    model_ridge = LinearRegression(input_dim=5, method='normal', regularization='ridge', alpha=1.0)
    model_ridge.fit(X_train_r, y_train_r)
    
    print(f"\n普通线性回归系数: {model_lr.get_weights()[0]}")
    print(f"岭回归系数 (alpha=1): {model_ridge.get_weights()[0]}")
    print(f"普通线性回归 R²: {model_lr.score(X_test_r, y_test_r):.4f}")
    print(f"岭回归 R²: {model_ridge.score(X_test_r, y_test_r):.4f}")
    
    # ========================================
    # 可视化
    # ========================================
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 示例 1: 简单线性回归
        axes[0].scatter(X_train.flatten(), y_train, alpha=0.5, label='Training Data')
        x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        axes[0].plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {weights[0]:.2f}x + {bias:.2f}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('y')
        axes[0].set_title('Simple Linear Regression')
        axes[0].legend()
        
        # 示例 2: 残差图
        predictions = model_multi.predict(X_test_m)
        residuals = y_test_m - predictions
        axes[1].scatter(predictions, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot (Multi-variate)')
        
        # 训练损失曲线
        axes[2].plot(model.loss_history)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss Curve')
        
        plt.tight_layout()
        plt.savefig('linear_regression_demo.png', dpi=150)
        print("\n图表已保存到 linear_regression_demo.png")
    except Exception as e:
        print(f"\n注意: 无法生成可视化图表 ({e})")
    
    print("\n" + "="*60)
    print("线性回归示例完成!")
    print("="*60)
