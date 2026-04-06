"""
================================================================================
                    NumPy 深度学习 - 神经网络
================================================================================
"""

import numpy as np
from typing import List, Optional
from .layers import Layer, Linear, Dropout, BatchNorm2D
from .losses_dl import CrossEntropyLoss
from .optimizer import SGD, Adam


class NeuralNetwork:
    """
    神经网络模型
    
    支持:
        - 任意层数的全连接网络
        - 多种激活函数
        - Dropout, BatchNorm
        - 反向传播训练
    """
    
    def __init__(self, layers: List[Layer], loss=None, optimizer=None):
        """
        参数:
            layers: 网络层列表
            loss: 损失函数
            optimizer: 优化器
        """
        self.layers = layers
        self.loss = loss if loss is not None else CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else SGD(learning_rate=0.01)
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = self.training
            elif isinstance(layer, BatchNorm2D):
                layer.training = self.training
            x = layer(x)
        return x
    
    def backward(self, grad_output: np.ndarray):
        """反向传播"""
        grad = grad_output
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
    
    def step(self):
        """更新参数"""
        self.optimizer.step(self.layers)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, verbose: bool = True):
        """
        训练模型
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 迭代次数
            batch_size: 批次大小
            verbose: 是否打印训练过程
        """
        n_samples = len(X_train)
        loss_history = []
        
        if verbose:
            print(f"\n{'='*60}")
            print("训练神经网络")
            print(f"{'='*60}")
            print(f"样本数: {n_samples}, 特征数: {X_train.shape[1]}")
            print(f"Epochs: {epochs}, Batch Size: {batch_size}")
            print(f"层数: {len(self.layers)}")
        
        for epoch in range(epochs):
            self.train()
            
            # 打乱数据
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # 前向传播
                predictions = self.forward(X_batch)
                
                # 计算损失
                loss = self.loss(predictions, y_batch)
                total_loss += loss
                n_batches += 1
                
                # 反向传播
                grad = self.loss.backward(predictions, y_batch)
                self.backward(grad)
                
                # 更新参数
                self.step()
            
            avg_loss = total_loss / n_batches
            loss_history.append(avg_loss)
            
            # 验证
            val_loss = 0.0
            if X_val is not None:
                self.eval()
                val_predictions = self.forward(X_val)
                val_loss = self.loss(val_predictions, y_val)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                val_info = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                train_acc = self.evaluate(X_train, y_train)
                val_info += f", Train Acc: {train_acc:.4f}"
                print(f"Epoch {epoch+1:5d}/{epochs} | Loss: {avg_loss:.4f}{val_info}")
        
        if verbose:
            print(f"\n训练完成!")
        
        return loss_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        self.eval()
        return self.forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def summary(self):
        """打印网络结构"""
        print(f"\n{'='*60}")
        print("神经网络结构")
        print(f"{'='*60}")
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                act = layer.activation.__class__.__name__ if layer.activation else 'None'
                print(f"Layer {i+1}: Linear({layer.input_size} -> {layer.output_size}), Activation: {act}")
            else:
                print(f"Layer {i+1}: {layer.__class__.__name__}")
        
        print(f"{'='*60}\n")


def create_mlp(input_dim: int, hidden_dims: List[int], output_dim: int,
               activation: str = 'relu', dropout: float = 0.0) -> NeuralNetwork:
    """创建多层感知机"""
    layers = []
    
    dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(dims) - 2):
        layers.append(Linear(dims[i], dims[i+1], activation=activation))
        if dropout > 0:
            layers.append(Dropout(dropout))
    
    # 输出层
    layers.append(Linear(dims[-2], dims[-1]))
    
    return NeuralNetwork(layers)


if __name__ == "__main__":
    print("="*60)
    print("NumPy 神经网络示例")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y[i, int(np.sum(X[i, :3] > 0)) % n_classes] = 1
    
    y_labels = np.argmax(y, axis=1)
    
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_labels[:split], y_labels[split:]
    
    # 创建网络
    model = create_mlp(input_dim=n_features, hidden_dims=[64, 32], 
                       output_dim=n_classes, activation='relu', dropout=0.2)
    
    model.summary()
    
    # 设置优化器
    model.optimizer = Adam(learning_rate=0.001)
    
    # 训练
    history = model.fit(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
    
    # 评估
    train_acc = model.evaluate(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)
    
    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    print("\n" + "="*60)
