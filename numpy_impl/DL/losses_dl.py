"""
================================================================================
                    NumPy 深度学习 - 损失函数
================================================================================
"""

import numpy as np


class Loss:
    """损失函数基类"""
    def __call__(self, predictions, targets):
        raise NotImplementedError
    
    def backward(self, predictions, targets):
        raise NotImplementedError


class MSELoss(Loss):
    """均方误差损失"""
    def __call__(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)
    
    def backward(self, predictions, targets):
        return 2 * (predictions - targets) / len(predictions)


class BCELoss(Loss):
    """二元交叉熵损失"""
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon
    
    def __call__(self, predictions, targets):
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    def backward(self, predictions, targets):
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        return -((targets / predictions) - ((1 - targets) / (1 - predictions))) / len(predictions)


class CrossEntropyLoss(Loss):
    """交叉熵损失"""
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon
    
    def __call__(self, predictions, targets):
        if len(targets.shape) == 1:
            n_samples = len(targets)
            n_classes = predictions.shape[1]
            targets_onehot = np.zeros((n_samples, n_classes))
            targets_onehot[np.arange(n_samples), targets] = 1
            targets = targets_onehot
        
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        return -np.sum(targets * np.log(predictions)) / len(predictions)
    
    def backward(self, predictions, targets):
        if len(targets.shape) == 1:
            n_samples = len(targets)
            n_classes = predictions.shape[1]
            targets_onehot = np.zeros((n_samples, n_classes))
            targets_onehot[np.arange(n_samples), targets] = 1
            targets = targets_onehot
        
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        return -(targets - predictions) / len(predictions)


if __name__ == "__main__":
    print("="*60)
    print("损失函数示例")
    print("="*60)
    
    np.random.seed(42)
    
    # 测试分类
    preds = np.random.rand(10, 3)
    preds = preds / preds.sum(axis=1, keepdims=True)
    targets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    
    ce_loss = CrossEntropyLoss()
    loss = ce_loss(preds, targets)
    grad = ce_loss.backward(preds, targets)
    
    print(f"\n交叉熵损失: {loss:.4f}")
    print(f"梯度形状: {grad.shape}")
    
    # 测试回归
    preds_reg = np.random.rand(10, 1)
    targets_reg = np.random.rand(10, 1)
    
    mse_loss = MSELoss()
    loss_reg = mse_loss(preds_reg, targets_reg)
    
    print(f"\nMSE 损失: {loss_reg:.4f}")
    
    print("\n" + "="*60)
