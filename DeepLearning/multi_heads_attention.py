import numpy as np

def softmax(x, axis=-1):
    """ 计算softmax函数 """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(q, k, v, mask):
    """ 计算缩放点积注意力 """
    matmul_qk = np.matmul(q, k.transpose((0, 1, 3, 2)))  # (..., seq_len_q, seq_len_k)
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    # 注意力权重不需要转置，因为它们已经是正确的形状
    # 执行矩阵乘法
    output = np.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return output, attention_weights

def multi_head_attention(queries, keys, values, num_heads, head_dim, mask):
    """ 多头注意力机制 """
    batch_size, seq_len, dim = queries.shape
    assert dim == num_heads * head_dim, "输入维度必须等于头数乘以头的维度"
    
    # 线性变换生成Q, K, V，这里不需要线性变换，直接重塑形状
    q = queries.reshape(batch_size, seq_len, num_heads, head_dim)
    k = keys.reshape(batch_size, seq_len, num_heads, head_dim)
    v = values.reshape(batch_size, seq_len, num_heads, head_dim)
    
    # 多头注意力
    attention_outputs, _ = scaled_dot_product_attention(q, k, v, mask)
    
    # 合并头，重塑形状
    attention_outputs = attention_outputs.reshape(batch_size, seq_len, num_heads * head_dim)
    
    return attention_outputs

# 示例输入
queries = np.random.rand(1, 10, 64)  # (batch_size, seq_len, dim)
keys = np.random.rand(1, 10, 64)    # 同queries
values = np.random.rand(1, 10, 64)   # 同queries

# 多头注意力参数
num_heads = 8
head_dim = 64 // num_heads

# 调用MHA
output = multi_head_attention(queries, keys, values, num_heads, head_dim, None)

print(output.shape)  # 应该与queries形状相同