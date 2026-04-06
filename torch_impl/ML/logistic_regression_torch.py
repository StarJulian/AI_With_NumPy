"""
================================================================================
                    PyTorch 逻辑回归
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LogisticRegressionTorch(nn.Module):
    """
    PyTorch 逻辑回归模型
    """
    
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if num_classes == 2:
            self.linear = nn.Linear(input_dim, 1)
        else:
            self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            return torch.sigmoid(self.linear(x)).squeeze()
        else:
            return self.linear(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            if self.num_classes == 2:
                return (output > 0.5).long()
            else:
                return output.argmax(dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            if self.num_classes == 2:
                probs = torch.stack([1 - output, output], dim=1)
            else:
                probs = torch.softmax(output, dim=1)
            return probs


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                              num_classes: int = 2,
                              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                              epochs: int = 1000, lr: float = 0.1,
                              verbose: bool = True) -> LogisticRegressionTorch:
    """
    训练逻辑回归模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    
    if num_classes == 2:
        y_train_t = torch.FloatTensor(y_train).to(device)
    else:
        y_train_t = torch.LongTensor(y_train).to(device)
    
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val).to(device)
        if num_classes == 2:
            y_val_t = torch.FloatTensor(y_val).to(device)
        else:
            y_val_t = torch.LongTensor(y_val).to(device)
    
    model = LogisticRegressionTorch(X_train.shape[1], num_classes).to(device)
    
    if num_classes == 2:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
        print(f"\n{'='*60}")
        print("训练 PyTorch 逻辑回归")
        print(f"{'='*60}")
        print(f"设备: {device}")
        print(f"样本数: {len(X_train)}, 类别数: {num_classes}")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                train_acc = calculate_accuracy(model, X_train_t, y_train_t, num_classes)
                val_info = ""
                if X_val is not None:
                    val_acc = calculate_accuracy(model, X_val_t, y_val_t, num_classes)
                    val_info = f", Val Acc: {val_acc:.4f}"
                print(f"Epoch {epoch+1:5d}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}{val_info}")
    
    if verbose:
        print(f"\n训练完成!")
    
    return model


def calculate_accuracy(model, X, y_true, num_classes):
    """计算准确率"""
    model.eval()
    with torch.no_grad():
        predictions = model.predict(X)
        if num_classes == 2:
            y_true = y_true.long()
        return (predictions == y_true).float().mean().item()


if __name__ == "__main__":
    print("="*60)
    print("PyTorch 逻辑回归示例")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 300
    
    # 二分类数据
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 训练
    model = train_logistic_regression(X_train, y_train, num_classes=2,
                                     X_val=X_test, y_val=y_test, epochs=200)
    
    # 测试
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    test_acc = calculate_accuracy(model, X_test_t, y_test_t, 2)
    
    print(f"\n测试集准确率: {test_acc:.4f}")
    
    print("\n" + "="*60)
