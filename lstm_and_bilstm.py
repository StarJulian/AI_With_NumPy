import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 参数初始化
        self.W_f = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_f = np.zeros((hidden_size, 1))
        
        self.W_i = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_i = np.zeros((hidden_size, 1))
        
        self.W_c = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_c = np.zeros((hidden_size, 1))
        
        self.W_o = np.random.randn(hidden_size, hidden_size + input_size)
        self.b_o = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        # x shape: (batch_size, seq_length, input_size)
        batch_size, seq_length, input_size = x.shape
        
        # 初始化输出
        h_t = np.zeros((batch_size, self.hidden_size))
        c_t = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_length):
            x_t = x[:, t, :]  # 当前时间步的输入
            combined = np.hstack((h_prev, x_t))  # 拼接输入和隐藏状态

            # 遗忘门
            f_t = sigmoid(np.dot(self.W_f, combined.T) + self.b_f)
            
            # 输入门
            i_t = sigmoid(np.dot(self.W_i, combined.T) + self.b_i)
            
            # 候选记忆细胞
            c_hat_t = tanh(np.dot(self.W_c, combined.T) + self.b_c)
            
            # 新的细胞状态
            c_t = f_t * c_prev.T + i_t * c_hat_t
            
            # 输出门
            o_t = sigmoid(np.dot(self.W_o, combined.T) + self.b_o)
            
            # 新的隐藏状态
            h_t = o_t * tanh(c_t)
            
            # 更新上一时间步的隐藏状态和细胞状态
            h_prev = h_t.T
            c_prev = c_t.T

        return h_t.T, c_t.T

# BiLSTM 类
class BiLSTM:
    def __init__(self, input_size, hidden_size):
        self.forward_lstm = LSTM(input_size, hidden_size)
        self.backward_lstm = LSTM(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, input_size = x.shape
        h_forward = np.zeros((batch_size, seq_length, self.forward_lstm.hidden_size))
        h_backward = np.zeros((batch_size, seq_length, self.backward_lstm.hidden_size))
        
        c_forward = np.zeros((batch_size, self.forward_lstm.hidden_size))
        c_backward = np.zeros((batch_size, self.backward_lstm.hidden_size))

        # 正向传播
        h_prev_forward = np.zeros((batch_size, self.forward_lstm.hidden_size))
        for t in range(seq_length):
            h_t, c_forward = self.forward_lstm.forward(x[:, t:t+1, :], h_prev_forward, c_forward)
            h_forward[:, t, :] = h_t
            h_prev_forward = h_t

        # 反向传播
        h_prev_backward = np.zeros((batch_size, self.backward_lstm.hidden_size))
        for t in reversed(range(seq_length)):
            h_t, c_backward = self.backward_lstm.forward(x[:, t:t+1, :], h_prev_backward, c_backward)
            h_backward[:, t, :] = h_t
            h_prev_backward = h_t

        # 将正向和反向的隐藏状态连接起来
        h_bidirectional = np.concatenate((h_forward, h_backward), axis=2)

        return h_bidirectional
    
import numpy as np

# 定义LSTM和BiLSTM类（假设已经定义在前面）

# 数据参数
batch_size = 2
seq_length = 5
input_size = 3
hidden_size = 4

# 生成随机输入数据
np.random.seed(0)
x = np.random.randn(batch_size, seq_length, input_size)

# 初始化隐藏状态和细胞状态为零
h_prev_v = np.zeros((batch_size, hidden_size))
c_prev_v = np.zeros((batch_size, hidden_size))

lstm = LSTM(input_size,hidden_size)
h_cur_v,c_cur_v = lstm.forward(x,h_prev_v,c_prev_v)

print("lstm输出:\n", h_cur_v.shape,c_cur_v.shape)

# 初始化BiLSTM
bilstm = BiLSTM(input_size, hidden_size)
# 前向传播
outputs = bilstm.forward(x)
# 打印输出
# print("输入:\n", x)
print("bilstm输出:\n", outputs.shape)


