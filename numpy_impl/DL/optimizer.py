"""
================================================================================
                    NumPy 深度学习 - 优化器
================================================================================
"""

import numpy as np
from typing import List


class Optimizer:
    """优化器基类"""
    def step(self, layers: List):
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def step(self, layers: List):
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                if self.momentum > 0:
                    if i not in self.velocity:
                        self.velocity[i] = {
                            'weights': np.zeros_like(layer.weights),
                            'bias': np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None
                        }
                    
                    self.velocity[i]['weights'] = self.momentum * self.velocity[i]['weights'] - \
                                                  self.learning_rate * layer.weights
                    layer.weights += self.velocity[i]['weights']
                    
                    if layer.bias is not None:
                        self.velocity[i]['bias'] = self.momentum * self.velocity[i]['bias'] - \
                                                    self.learning_rate * layer.bias
                        layer.bias += self.velocity[i]['bias']
                else:
                    layer.weights -= self.learning_rate * layer.weights
                    if layer.bias is not None:
                        layer.bias -= self.learning_rate * layer.bias


class Adam(Optimizer):
    """Adam 优化器"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
    
    def step(self, layers: List):
        self.t += 1
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                if i not in self.m:
                    self.m[i] = {'weights': np.zeros_like(layer.weights)}
                    self.v[i] = {'weights': np.zeros_like(layer.weights)}
                
                # 更新一阶和二阶矩估计
                self.m[i]['weights'] = self.beta1 * self.m[i]['weights'] + (1 - self.beta1) * layer.weights
                self.v[i]['weights'] = self.beta2 * self.v[i]['weights'] + (1 - self.beta2) * (layer.weights ** 2)
                
                # 偏差校正
                m_hat = self.m[i]['weights'] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i]['weights'] / (1 - self.beta2 ** self.t)
                
                # 更新参数
                layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                if hasattr(layer, 'bias') and layer.bias is not None:
                    if 'bias' not in self.m[i]:
                        self.m[i]['bias'] = np.zeros_like(layer.bias)
                        self.v[i]['bias'] = np.zeros_like(layer.bias)
                    
                    self.m[i]['bias'] = self.beta1 * self.m[i]['bias'] + (1 - self.beta1) * layer.bias
                    self.v[i]['bias'] = self.beta2 * self.v[i]['bias'] + (1 - self.beta2) * (layer.bias ** 2)
                    
                    m_hat_b = self.m[i]['bias'] / (1 - self.beta1 ** self.t)
                    v_hat_b = self.v[i]['bias'] / (1 - self.beta2 ** self.t)
                    
                    layer.bias -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


class Momentum(Optimizer):
    """动量优化器"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def step(self, layers: List):
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                if i not in self.velocity:
                    self.velocity[i] = {
                        'weights': np.zeros_like(layer.weights),
                        'bias': np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None
                    }
                
                self.velocity[i]['weights'] = self.momentum * self.velocity[i]['weights'] - \
                                              self.learning_rate * layer.weights
                layer.weights += self.velocity[i]['weights']
                
                if layer.bias is not None:
                    self.velocity[i]['bias'] = self.momentum * self.velocity[i]['bias'] - \
                                                self.learning_rate * layer.bias
                    layer.bias += self.velocity[i]['bias']


if __name__ == "__main__":
    print("="*60)
    print("优化器示例")
    print("="*60)
    
    # 创建一些虚拟层
    class DummyLayer:
        def __init__(self):
            self.weights = np.random.randn(5, 3)
            self.bias = np.random.randn(1, 3)
    
    layers = [DummyLayer() for _ in range(3)]
    
    print(f"\n初始权重 (第一层): {layers[0].weights[0]}")
    
    # SGD
    sgd = SGD(learning_rate=0.1)
    sgd.step(layers)
    print(f"SGD 更新后: {layers[0].weights[0]}")
    
    # Adam
    layers2 = [DummyLayer() for _ in range(3)]
    adam = Adam(learning_rate=0.1)
    adam.step(layers2)
    print(f"Adam 更新后: {layers2[0].weights[0]}")
    
    print("\n" + "="*60)
