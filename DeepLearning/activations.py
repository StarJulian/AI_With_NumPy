"""
激活函数模块
提供各种常用的激活函数实现
"""

import numpy as np


class Sigmoid:
    """Sigmoid 激活函数"""
    
    def __call__(self, x):
        """前向传播"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def gradient(self, x):
        """梯度计算"""
        s = self(x)
        return s * (1 - s)


class Tanh:
    """Tanh 激活函数"""
    
    def __call__(self, x):
        """前向传播"""
        return np.tanh(x)
    
    def gradient(self, x):
        """梯度计算"""
        return 1 - np.tanh(x) ** 2


class ReLU:
    """ReLU (Rectified Linear Unit) 激活函数"""
    
    def __init__(self, alpha=0.0):
        """
        参数:
            alpha: Leaky ReLU 的负斜率，默认 0.0 即标准 ReLU
        """
        self.alpha = alpha
    
    def __call__(self, x):
        """前向传播"""
        return np.where(x > 0, x, self.alpha * x)
    
    def gradient(self, x):
        """梯度计算"""
        return np.where(x > 0, 1, self.alpha)


class LeakyReLU(ReLU):
    """Leaky ReLU 激活函数"""
    
    def __init__(self, alpha=0.01):
        super().__init__(alpha=alpha)


class ELU:
    """ELU (Exponential Linear Unit) 激活函数"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x):
        """前向传播"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def gradient(self, x):
        """梯度计算"""
        return np.where(x > 0, 1, self.alpha * np.exp(x))


class GELU:
    """GELU (Gaussian Error Linear Unit) 激活函数
    近似于: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    
    def __call__(self, x):
        """前向传播"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def gradient(self, x):
        """梯度计算（近似）"""
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
        return cdf + x * pdf


class SiLU(Sigmoid):
    """SiLU / Swish 激活函数: x * sigmoid(x)"""
    
    def __call__(self, x):
        s = super().__call__(x)
        return x * s
    
    def gradient(self, x):
        s = super().__call__(x)
        return s + x * s * (1 - s)


class Softmax:
    """Softmax 激活函数（多分类输出层）"""
    
    def __call__(self, x, axis=-1):
        """前向传播
        
        参数:
            x: 输入数据
            axis: 沿着哪个轴计算 softmax
        """
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def gradient(self, x):
        """梯度计算"""
        # Softmax 的梯度较复杂，这里返回简化的梯度
        # 实际应用中通常使用 CrossEntropyLoss 的组合梯度
        p = self(x)
        return p * (1 - p)


class Softplus:
    """Softplus 激活函数: log(1 + exp(x))"""
    
    def __call__(self, x):
        """前向传播"""
        return np.log(1 + np.exp(np.clip(x, -500, 500)))
    
    def gradient(self, x):
        """梯度计算"""
        return 1 / (1 + np.exp(-x))


def get_activation(name):
    """获取激活函数实例
    
    参数:
        name: 激活函数名称
        
    返回:
        激活函数实例
    """
    activations = {
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'relu': ReLU(),
        'leaky_relu': LeakyReLU(),
        'elu': ELU(),
        'gelu': GELU(),
        'silu': SiLU(),
        'swish': SiLU(),
        'softmax': Softmax(),
        'softplus': Softplus(),
    }
    
    if isinstance(name, str):
        name = name.lower()
        if name not in activations:
            raise ValueError(f"未知的激活函数: {name}")
        return activations[name]
    return name


# 导出所有激活函数
__all__ = [
    'Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU', 'ELU', 'GELU', 'SiLU',
    'Softmax', 'Softplus', 'get_activation'
]
