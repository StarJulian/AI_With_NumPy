"""
================================================================================
                    PyTorch 线性回归
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LinearRegressionTorch(nn.Module):
    """
    PyTorch 线性回归模型
    """
    
    def __init__(self, input_dim: int, regularization: Optional[str] = None, 
                 alpha: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)
        self.regularization = regularization
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.linear.weight.data, self.linear.bias.data


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                            epochs: int = 1000, lr: float = 0.01,
                            regularization: Optional[str] = None, alpha: float = 0.01,
                            verbose: bool = True) -> LinearRegressionTorch:
    """
    训练线性回归模型
    
    参数:
        X_train: 训练数据
        y_train: 目标值
        X_val: 验证数据
        y_val: 验证目标值
        epochs: 迭代次数
        lr: 学习率
        regularization: 正则化类型 ('l2')
        alpha: 正则化强度
        verbose: 是否打印训练过程
    
    返回:
        训练好的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 转换为张量
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
    
    # 创建模型
    model = LinearRegressionTorch(X_train.shape[1], regularization, alpha).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
        print(f"\n{'='*60}")
        print("训练 PyTorch 线性回归")
        print(f"{'='*60}")
        print(f"设备: {device}")
        print(f"样本数: {len(X_train)}, 特征数: {X_train.shape[1]}")
        print(f"学习率: {lr}, 迭代次数: {epochs}")
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        # 添加正则化
        if regularization == 'l2':
            l2_loss = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + alpha * l2_loss
        
        loss.backward()
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            history['val_loss'].append(val_loss)
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            val_info = f", Val Loss: {val_loss:.6f}" if X_val is not None else ""
            print(f"Epoch {epoch+1:5d}/{epochs} | Loss: {loss.item():.6f}{val_info}")
    
    if verbose:
        print(f"\n训练完成!")
        weights, bias = model.get_weights()
        print(f"权重: {weights.squeeze().cpu().numpy()[:5]}...")
        print(f"偏置: {bias.item():.6f}")
    
    return model


if __name__ == "__main__":
    print("="*60)
    print("PyTorch 线性回归示例")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 3) * 5
    y = 1 + 2*X[:, 0] + 3*X[:, 1] + 4*X[:, 2] + np.random.randn(n_samples) * 2
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 训练
    model = train_linear_regression(X_train, y_train, X_test, y_test, epochs=500)
    
    # 评估
    X_test_t = torch.FloatTensor(X_test)
    predictions = model.predict(X_test_t)
    
    mse = np.mean((predictions.numpy() - y_test) ** 2)
    ss_res = np.sum((y_test - predictions.numpy()) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\n测试集 MSE: {mse:.4f}")
    print(f"测试集 R²: {r2:.4f}")
    
    print("\n" + "="*60)
