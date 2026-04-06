"""
================================================================================
                    NumPy 深度学习 - 激活函数
================================================================================
"""

import numpy as np


class Sigmoid:
    """Sigmoid 激活函数"""
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def gradient(self, x):
        s = self(x)
        return s * (1 - s)


class Tanh:
    """Tanh 激活函数"""
    def __call__(self, x):
        return np.tanh(x)
    
    def gradient(self, x):
        return 1 - np.tanh(x) ** 2


class ReLU:
    """ReLU 激活函数"""
    def __init__(self, alpha=0.0):
        self.alpha = alpha
    
    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def gradient(self, x):
        return np.where(x > 0, 1, self.alpha)


class LeakyReLU(ReLU):
    def __init__(self, alpha=0.01):
        super().__init__(alpha=alpha)


class GELU:
    """GELU 激活函数"""
    def __call__(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def gradient(self, x):
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
        return cdf + x * pdf


class Softmax:
    """Softmax 激活函数"""
    def __call__(self, x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def gradient(self, x):
        p = self(x)
        return p * (1 - p)


class SiLU:
    """SiLU / Swish: x * sigmoid(x)"""
    def __call__(self, x):
        s = 1.0 / (1.0 + np.exp(-x))
        return x * s
    
    def gradient(self, x):
        s = 1.0 / (1.0 + np.exp(-x))
        return s + x * s * (1 - s)


def get_activation(name):
    """获取激活函数"""
    activations = {
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'relu': ReLU(),
        'leaky_relu': LeakyReLU(),
        'gelu': GELU(),
        'silu': SiLU(),
        'swish': SiLU(),
        'softmax': Softmax(),
    }
    name = name.lower()
    return activations.get(name, ReLU())


if __name__ == "__main__":
    print("="*60)
    print("激活函数示例")
    print("="*60)
    
    x = np.linspace(-5, 5, 20)
    
    print(f"\nReLU: {ReLU()(x)[:5]}...")
    print(f"Tanh: {Tanh()(x)[:5]}...")
    print(f"Sigmoid: {Sigmoid()(x)[:5]}...")
    print(f"GELU: {GELU()(x)[:5]}...")
    print(f"Softmax (多维): {Softmax()(np.array([x, x*2]))[:, :5]}...")
    
    print("\n" + "="*60)
