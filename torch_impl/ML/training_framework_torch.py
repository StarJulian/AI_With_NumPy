"""
================================================================================
                    PyTorch 训练框架
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass
import time


@dataclass
class TrainingHistory:
    train_losses: List[float]
    val_losses: List[float]
    train_accuracies: List[float]
    val_accuracies: List[float]
    epochs: int
    total_time: float


class MLP(nn.Module):
    """
    多层感知机 / 全连接神经网络
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPTrainer:
    """MLP 训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.history: Optional[TrainingHistory] = None
    
    def train_epoch(self, train_loader, criterion, optimizer) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y_batch)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        
        return total_loss / total, correct / total
    
    def evaluate(self, val_loader, criterion) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                total_loss += loss.item() * len(y_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        
        return total_loss / total, correct / total
    
    def fit(self, train_loader, val_loader=None, epochs=100, lr=0.001,
            weight_decay=0.0, verbose=True) -> TrainingHistory:
        """训练模型"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        if verbose:
            print(f"\n{'='*60}")
            print("训练 MLP")
            print(f"{'='*60}")
            print(f"设备: {self.device}")
            print(f"Epochs: {epochs}, Learning Rate: {lr}")
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = 0, 0
            
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            scheduler.step()
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                val_info = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if val_loader else ""
                print(f"Epoch {epoch+1:5d}/{epochs} | "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}{val_info}")
        
        total_time = time.time() - start_time
        
        self.history = TrainingHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accuracies=train_accs,
            val_accuracies=val_accs,
            epochs=epochs,
            total_time=total_time
        )
        
        if verbose:
            print(f"\n训练完成! 用时: {total_time:.2f}秒")
        
        return self.history
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """推理"""
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            return self.model(X)


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                batch_size: int = 32, epochs: int = 100, lr: float = 0.001,
                verbose: bool = True) -> TrainingHistory:
    """快速训练模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    trainer = MLPTrainer(model, device='auto')
    return trainer.fit(train_loader, val_loader, epochs=epochs, lr=lr, verbose=verbose)


def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                   batch_size: int = 32) -> Dict[str, float]:
    """评估模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    
    trainer = MLPTrainer(model)
    test_loss, test_acc = trainer.evaluate(test_loader, criterion)
    
    return {'loss': test_loss, 'accuracy': test_acc}


if __name__ == "__main__":
    print("="*60)
    print("PyTorch 训练框架示例")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 创建模型
    model = MLP(input_dim=n_features, hidden_dims=[64, 32], output_dim=n_classes, dropout=0.2)
    
    # 训练
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=100)
    
    # 评估
    results = evaluate_model(model, X_test, y_test)
    print(f"\n测试集损失: {results['loss']:.4f}")
    print(f"测试集准确率: {results['accuracy']:.4f}")
    
    print("\n" + "="*60)
