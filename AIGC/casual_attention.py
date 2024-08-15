import numpy as np

def softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # 减去最大值以提高数值稳定性
    shift_x = x - np.max(x, axis=-1, keepdims=True)
    # 计算e的指数
    exp_x = np.exp(shift_x)
    # 归一化，使和为1
    softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return softmax_x

def causal_self_attention(Q, K, V, mask):
    dim_key = K.shape[-1]
    attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / (np.sqrt(dim_key) + 1e-9)
    attention_scores = np.where(mask == 0, -np.inf, attention_scores)
    attention_scores = np.nan_to_num(attention_scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    
    # 使用数值稳定的softmax
    attention_weights = softmax(attention_scores)
    
    # 确保无效值处理后不会影响计算结果
    attention_weights = np.nan_to_num(attention_weights, nan=0.0, posinf=0.0, neginf=0.0)
    
    output = np.matmul(attention_weights, V)
    return output

# 示例用法
batch_size = 2
seq_length = 4
dim = 8

Q = np.random.rand(batch_size, seq_length, dim)
K = np.random.rand(batch_size, seq_length, dim)
V = np.random.rand(batch_size, seq_length, dim)

# 创建一个上三角掩码矩阵
mask = np.triu(np.ones((seq_length, seq_length)), k=1)[np.newaxis, np.newaxis, :, :]

# 调用causal_self_attention函数
output = causal_self_attention(Q, K, V, mask)
print(output)
