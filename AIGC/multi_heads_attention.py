import numpy as np

def softmax(x, axis=-1):
    exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)

def scaled_dot_product_attention(q, k, v, scale, mask=None):
    # q, k, v 的形状应为 (batch_size, num_heads, seq_len, head_dim)
    # 计算 q 和 k 的点积，并缩放
    matmul_qk = np.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # 转置 k 的最后两个维度

    if mask is not None:
        matmul_qk = np.where(mask == 0, -1e9, matmul_qk)  # 应用掩码

    # 计算注意力权重
    attention_weights = softmax(matmul_qk, axis=-1)

    # 计算最终的注意力输出
    output = np.matmul(attention_weights, v)
    return output, attention_weights

def multi_head_attention(queries, keys, values, num_heads, head_dim, mask=None):
    batch_size, seq_len, d_model = queries.shape
    assert d_model == num_heads * head_dim, "d_model 必须等于 num_heads * head_dim"

    # 线性变换并拆分为多头
    q = queries.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = keys.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = values.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # 计算缩放点积注意力
    scale = head_dim ** 0.5
    attention_output, _ = scaled_dot_product_attention(q, k, v, scale, mask)

    # 合并所有头的输出
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    return attention_output


batch_size = 2
seq_len = 5
d_model = 16  # 总的特征维度
num_heads = 4
head_dim = d_model // num_heads  # 每个头的特征维度

queries = np.random.rand(batch_size, seq_len, d_model)
keys = np.random.rand(batch_size, seq_len, d_model)
values = np.random.rand(batch_size, seq_len, d_model)


# 应用多头注意力机制
output = multi_head_attention(queries, keys, values, num_heads, head_dim)

print("输入形状:", queries.shape)
print("输出形状:", output.shape)


