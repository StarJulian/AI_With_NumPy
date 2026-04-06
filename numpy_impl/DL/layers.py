"""
================================================================================
                    NumPy 深度学习 - 层实现
================================================================================
"""

import numpy as np
from typing import Optional
from .activations import get_activation


class Layer:
    """层基类"""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)


class Linear(Layer):
    """全连接层"""
    
    def __init__(self, input_size: int, output_size: int, activation=None):
        self.input_size = input_size
        self.output_size = output_size
        
        # Xavier 初始化
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros((1, output_size))
        
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.input_cache = None
        self.output_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        self.output = np.dot(x, self.weights) + self.bias
        
        if self.activation is not None:
            self.output_cache = self.output
            self.output = self.activation(self.output)
        
        return self.output
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        if self.activation is not None and self.output_cache is not None:
            grad_output = grad_output * self.activation.gradient(self.output_cache)
        
        grad_weights = np.dot(self.input_cache.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # 更新参数
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input


class Conv2D(Layer):
    """2D 卷积层 (简化版)"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He 初始化
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros((1, out_channels, 1, 1))
        
        self.input_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        batch_size, in_c, h, w = x.shape
        
        if self.padding > 0:
            x = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        
        out_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1
        
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        region = x[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        out[b, oc, oh, ow] = np.sum(region * self.weights[oc]) + self.bias[0, oc, 0, 0]
        
        return out
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        return np.zeros_like(self.input_cache)


class MaxPool2D(Layer):
    """2D 最大池化层"""
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.input_cache = None
        self.max_indices = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        batch_size, channels, h, w = x.shape
        
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        
        out = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros_like(out, dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        region = x[b, c, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                        out[b, c, oh, ow] = np.max(region)
        
        return out
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        return np.zeros_like(self.input_cache)


class Dropout(Layer):
    """Dropout 层"""
    
    def __init__(self, drop_prob: float = 0.5):
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.rand(*x.shape) < self.keep_prob
            return x * self.mask / self.keep_prob
        return x
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        return grad_output * self.mask / self.keep_prob
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class BatchNorm2D(Layer):
    """Batch Normalization 层"""
    
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        return self.gamma * x_normalized + self.beta
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        return grad_output
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class LayerNorm2D(Layer):
    """Layer Normalization 层"""
    
    def __init__(self, num_features: int, epsilon: float = 1e-5):
        self.num_features = num_features
        self.epsilon = epsilon
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
        var = np.var(x, axis=(1, 2, 3), keepdims=True)
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_normalized + self.beta
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        return grad_output


if __name__ == "__main__":
    print("="*60)
    print("NumPy 层示例")
    print("="*60)
    
    # 测试 Linear 层
    print("\n全连接层:")
    linear = Linear(10, 5, activation='relu')
    x = np.random.randn(3, 10)
    y = linear(x)
    print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 测试 Dropout
    print("\nDropout:")
    dropout = Dropout(0.5)
    dropout.train()
    y_drop = dropout(x)
    dropout.eval()
    y_eval = dropout(x)
    print(f"训练模式: {y_drop}")
    print(f"评估模式: {y_eval}")
    
    # 测试 BatchNorm
    print("\nBatchNorm:")
    bn = BatchNorm2D(3)
    x_bn = np.random.randn(2, 3, 4, 4)
    y_bn = bn(x_bn)
    print(f"输入形状: {x_bn.shape}, 输出形状: {y_bn.shape}")
    
    print("\n" + "="*60)
