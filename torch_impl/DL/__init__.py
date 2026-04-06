"""
================================================================================
                    PyTorch 深度学习模块
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict
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
    
    使用 PyTorch 实现，支持 GPU 加速
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if i < len(dims) - 2:  # 输出层前不加激活和dropout
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'silu':
                    layers.append(nn.SiLU())
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNN(nn.Module):
    """卷积神经网络"""
    
    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class RNN(nn.Module):
    """RNN 循环神经网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, 
                         dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        output, hidden = self.rnn(x)
        # 取最后一个时间步
        output = self.fc(output[:, -1, :])
        return output


class LSTMModel(nn.Module):
    """LSTM 长短期记忆网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output


class TransformerClassifier(nn.Module):
    """基于 Transformer 的分类器"""
    
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 平均池化
        return self.fc(x)


class Trainer:
    """PyTorch 训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.history: Optional[TrainingHistory] = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, lr: float = 0.001,
            weight_decay: float = 0.0, verbose: bool = True) -> TrainingHistory:
        """训练模型"""
        
        # 转换数据
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val)
            val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        
        if verbose:
            print(f"\n{'='*60}")
            print("训练 PyTorch 神经网络")
            print(f"{'='*60}")
            print(f"设备: {self.device}")
            print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
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
                
                train_loss += loss.item() * len(y_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
            
            train_loss /= len(train_dataset)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc = 0, 0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss_sum = 0
                    val_correct = 0
                    val_total = 0
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        
                        val_loss_sum += loss.item() * len(y_batch)
                        _, predicted = outputs.max(1)
                        val_total += y_batch.size(0)
                        val_correct += predicted.eq(y_batch).sum().item()
                    
                    val_loss = val_loss_sum / len(val_dataset)
                    val_acc = val_correct / val_total
                
                val_losses.append(val_loss)
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            return outputs.argmax(dim=1).cpu().numpy()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


if __name__ == "__main__":
    print("="*60)
    print("PyTorch 神经网络示例")
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
    
    # 创建并训练 MLP
    model = MLP(input_dim=n_features, hidden_dims=[128, 64, 32], 
                output_dim=n_classes, dropout=0.3)
    
    trainer = Trainer(model)
    history = trainer.fit(X_train, y_train, X_test, y_test, epochs=100)
    
    # 评估
    test_acc = trainer.score(X_test, y_test)
    print(f"\n测试集准确率: {test_acc:.4f}")
    
    print("\n" + "="*60)
