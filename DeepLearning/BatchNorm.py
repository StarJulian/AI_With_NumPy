import numpy as np

class BatchNorm2D:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        
    def forward(self, x, training=True):
        if training:
            # 计算每个通道的均值和方差
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            # 更新运行中的均值和方差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # 归一化输入
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            # 使用运行中的均值和方差进行归一化
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # 应用 gamma 和 beta
        out = self.gamma * x_normalized + self.beta
        return out
    
    def update_params(self, gamma, beta):
        self.gamma = gamma
        self.beta = beta

# 示例使用
# 输入数据 (batch_size, num_channels, height, width)
x = np.random.randn(5, 3, 32, 32)

# 初始化 BatchNorm2D 层
batch_norm2d = BatchNorm2D(num_features=3)

# 前向传播（训练模式）
output = batch_norm2d.forward(x, training=True)
print("训练模式下的输出：")
print(output)

# 前向传播（推理模式）
output = batch_norm2d.forward(x, training=False)
print("推理模式下的输出：")
print(output)
