import numpy as np

class LayerNorm2D:
    def __init__(self, num_features, epsilon=1e-5):
        self.num_features = num_features
        self.epsilon = epsilon
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        
    def forward(self, x):
        # 计算每个样本每个通道的均值和方差
        mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
        var = np.var(x, axis=(1, 2, 3), keepdims=True)
        
        # 归一化输入
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        
        # 应用 gamma 和 beta
        out = self.gamma * x_normalized + self.beta
        return out
    
    def update_params(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

# 示例使用
# 输入数据 (batch_size, num_channels, height, width)
x = np.random.randn(5, 3, 32, 32)

# 初始化 LayerNorm2D 层
layer_norm2d = LayerNorm2D(num_features=3)

# 前向传播
output = layer_norm2d.forward(x)
print("输出：")
print(output)
