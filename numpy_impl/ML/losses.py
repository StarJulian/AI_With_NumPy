"""
================================================================================
                        NumPy 机器学习 - 损失函数
================================================================================
"""

import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """损失函数基类"""
    
    @abstractmethod
    def __call__(self, predictions, targets):
        """计算损失"""
        pass
    
    @abstractmethod
    def backward(self, predictions, targets):
        """反向传播"""
        pass


class MeanSquaredError(Loss):
    """均方误差损失 (MSE)"""
    
    def __call__(self, predictions, targets):
        """
        计算 MSE 损失
        
        参数:
            predictions: 预测值 (n_samples, n_outputs)
            targets: 目标值 (n_samples,) 或 (n_samples, n_outputs)
        """
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        return np.mean((predictions - targets) ** 2)
    
    def backward(self, predictions, targets):
        """MSE 的梯度: 2 * (predictions - targets) / n"""
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        n = len(predictions)
        return 2 * (predictions - targets) / n


class CrossEntropyLoss(Loss):
    """交叉熵损失 (用于多分类)"""
    
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon
    
    def __call__(self, predictions, targets):
        """
        计算交叉熵损失
        
        参数:
            predictions: 预测概率 (n_samples, n_classes)，经过 softmax
            targets: 目标类别 (n_samples,) 或 one-hot 编码 (n_samples, n_classes)
        """
        # 确保预测值在有效范围内
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        # 转换为 one-hot 如果是类别索引
        if len(targets.shape) == 1:
            n_samples = len(targets)
            n_classes = predictions.shape[1]
            targets_onehot = np.zeros((n_samples, n_classes))
            targets_onehot[np.arange(n_samples), targets] = 1
            targets = targets_onehot
        
        # 计算交叉熵
        n_samples = len(predictions)
        loss = -np.sum(targets * np.log(predictions)) / n_samples
        return loss
    
    def backward(self, predictions, targets):
        """交叉熵的梯度（在 softmax 之后）"""
        if len(targets.shape) == 1:
            n_samples = len(targets)
            n_classes = predictions.shape[1]
            targets_onehot = np.zeros((n_samples, n_classes))
            targets_onehot[np.arange(n_samples), targets] = 1
            targets = targets_onehot
        
        n_samples = len(predictions)
        return -(targets - predictions) / n_samples


class BinaryCrossEntropyLoss(Loss):
    """二元交叉熵损失 (用于二分类)"""
    
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon
    
    def __call__(self, predictions, targets):
        """
        计算二元交叉熵损失
        
        参数:
            predictions: 预测概率 (n_samples,) 或 (n_samples, 1)
            targets: 目标值 (n_samples,) 或 (n_samples, 1)
        """
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        if len(predictions.shape) == 2:
            predictions = predictions.flatten()
        if len(targets.shape) == 2:
            targets = targets.flatten()
        
        n_samples = len(predictions)
        loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return loss
    
    def backward(self, predictions, targets):
        """二元交叉熵的梯度"""
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        if len(predictions.shape) == 2:
            predictions = predictions.flatten()
        if len(targets.shape) == 2:
            targets = targets.flatten()
        
        n_samples = len(predictions)
        return -((targets / predictions) - ((1 - targets) / (1 - predictions))) / n_samples


class HingeLoss(Loss):
    """Hinge 损失 (用于 SVM)"""
    
    def __call__(self, predictions, targets):
        """
        计算 Hinge 损失
        
        参数:
            predictions: 预测值 (n_samples,)
            targets: 目标值 (+1 或 -1)
        """
        return np.mean(np.maximum(0, 1 - targets * predictions))
    
    def backward(self, predictions, targets):
        """Hinge 损失的梯度"""
        grad = np.zeros_like(predictions)
        mask = (1 - targets * predictions) > 0
        grad[mask] = -targets[mask]
        return grad


class MAELoss(Loss):
    """平均绝对误差损失 (MAE)"""
    
    def __call__(self, predictions, targets):
        """计算 MAE 损失"""
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        return np.mean(np.abs(predictions - targets))
    
    def backward(self, predictions, targets):
        """MAE 的梯度"""
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        return np.sign(predictions - targets) / len(predictions)


def get_loss(name: str) -> Loss:
    """获取损失函数"""
    losses = {
        'mse': MeanSquaredError,
        'mae': MAELoss,
        'cross_entropy': CrossEntropyLoss,
        'bce': BinaryCrossEntropyLoss,
        'hinge': HingeLoss,
    }
    
    name = name.lower()
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}")
    
    return losses[name]()


# =============================================================================
#                           示例: 使用损失函数
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("损失函数示例")
    print("="*60)
    
    # 生成随机预测和目标
    np.random.seed(42)
    predictions = np.random.rand(100, 3)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)  # softmax 概率
    targets = np.random.randint(0, 3, size=100)
    
    # 测试交叉熵损失
    ce_loss = CrossEntropyLoss()
    loss_value = ce_loss(predictions, targets)
    print(f"\n交叉熵损失: {loss_value:.4f}")
    
    # 测试 MSE 损失
    mse_loss = MeanSquaredError()
    y_regression = np.random.rand(100, 1)
    mse_value = mse_loss(predictions[:, :1], y_regression)
    print(f"MSE 损失: {mse_value:.4f}")
    
    # 测试 BCE 损失
    bce_loss = BinaryCrossEntropyLoss()
    y_binary = np.random.randint(0, 2, size=100)
    p_binary = np.random.rand(100)
    bce_value = bce_loss(p_binary, y_binary)
    print(f"BCE 损失: {bce_value:.4f}")
