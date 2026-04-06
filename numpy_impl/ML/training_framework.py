"""
================================================================================
                    NumPy 机器学习 - 训练框架
================================================================================
提供统一的训练和推理接口
================================================================================
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class TrainingHistory:
    """训练历史记录"""
    train_losses: List[float]
    val_losses: List[float]
    train_accuracies: List[float]
    val_accuracies: List[float]
    epochs: int
    total_time: float


class Model:
    """模型基类"""
    
    def fit(self, X, y, **kwargs):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def save(self, path: str):
        """保存模型"""
        raise NotImplementedError
    
    def load(self, path: str):
        """加载模型"""
        raise NotImplementedError


class Trainer:
    """训练器类"""
    
    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = None
    
    def train_epoch(self, X_train, y_train, batch_size=32) -> Tuple[float, float]:
        """训练一个 epoch"""
        n_samples = len(X_train)
        indices = np.random.permutation(n_samples)
        total_loss = 0.0
        correct = 0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # 前向传播
            predictions = self.model.forward(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # 反向传播
            grad = self.criterion.backward(predictions, y_batch)
            self.model.backward(grad, self.optimizer.learning_rate)
            
            total_loss += loss * len(batch_indices)
            if hasattr(self.model, 'predict'):
                preds = np.argmax(predictions, axis=1)
                correct += np.sum(preds == y_batch)
        
        avg_loss = total_loss / n_samples
        accuracy = correct / n_samples
        return avg_loss, accuracy
    
    def evaluate(self, X_val, y_val, batch_size=32) -> Tuple[float, float]:
        """评估模型"""
        n_samples = len(X_val)
        total_loss = 0.0
        correct = 0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_val[start_idx:end_idx]
            y_batch = y_val[start_idx:end_idx]
            
            predictions = self.model.forward(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            total_loss += loss * len(X_batch)
            preds = np.argmax(predictions, axis=1)
            correct += np.sum(preds == y_batch)
        
        avg_loss = total_loss / n_samples
        accuracy = correct / n_samples
        return avg_loss, accuracy
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32, verbose=True) -> TrainingHistory:
        """训练模型"""
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # 验证
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
            else:
                val_losses.append(0)
                val_accuracies.append(0)
            
            # 打印进度
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                if X_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        total_time = time.time() - start_time
        
        self.history = TrainingHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            epochs=epochs,
            total_time=total_time
        )
        
        return self.history
    
    def predict(self, X) -> np.ndarray:
        """推理"""
        return self.model.forward(X)


class TrainingFramework:
    """训练框架 - 简化版 PyTorch Lightning 风格"""
    
    def __init__(self):
        self.models: Dict[str, Model] = {}
        self.history: Dict[str, TrainingHistory] = {}
    
    def register_model(self, name: str, model: Model):
        """注册模型"""
        self.models[name] = model
    
    def train(self, model_name: str, X_train, y_train, 
              X_val=None, y_val=None, epochs=100, batch_size=32) -> TrainingHistory:
        """训练指定模型"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        trainer = Trainer(model, None, None)
        history = trainer.fit(X_train, y_train, X_val, y_val, epochs, batch_size)
        self.history[model_name] = history
        return history
    
    def predict(self, model_name: str, X) -> np.ndarray:
        """使用指定模型推理"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name].forward(X)
    
    def summary(self):
        """打印模型摘要"""
        print("\n" + "="*60)
        print("模型摘要 / Model Summary")
        print("="*60)
        for name, model in self.models.items():
            print(f"\n模型: {name}")
            if hasattr(model, '__str__'):
                print(str(model))
        print("="*60)


# =============================================================================
#                           示例: 使用训练框架
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("NumPy 机器学习训练框架示例")
    print("="*60)
    
    # 生成示例数据
    np.random.seed(42)
    
    # 分类数据
    n_samples = 500
    X_class = np.random.randn(n_samples, 2)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)
    
    # 划分训练集和验证集
    split = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, X_val = X_class[train_idx], X_class[val_idx]
    y_train, y_val = y_class[train_idx], y_class[val_idx]
    
    print(f"\n训练样本数: {len(X_train)}")
    print(f"验证样本数: {len(X_val)}")
    print(f"类别分布: {np.bincount(y_train)}")
    
    # 使用训练框架
    from .logistic_regression import LogisticRegression
    from .losses import CrossEntropyLoss
    
    # 创建模型
    model = LogisticRegression(input_dim=2, num_classes=2)
    criterion = CrossEntropyLoss()
    optimizer = {'learning_rate': 0.1}
    
    # 创建训练器
    trainer = Trainer(model, criterion, optimizer)
    
    # 训练
    print("\n开始训练 Logistic Regression...")
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # 评估
    train_loss, train_acc = trainer.evaluate(X_train, y_train)
    val_loss, val_acc = trainer.evaluate(X_val, y_val)
    
    print(f"\n最终结果:")
    print(f"训练集 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"验证集 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    print(f"总训练时间: {history.total_time:.2f}秒")
