"""
================================================================================
                        ASR 模型 - NumPy 实现
================================================================================
包含:
    - CTC: 连接时序分类
    - 简化音频特征提取
================================================================================
"""

import numpy as np
from typing import List, Tuple, Optional


# =============================================================================
#                           音频处理工具
# =============================================================================
def extract_mfcc(audio: np.ndarray, sample_rate: int = 16000,
                 n_mfcc: int = 13, n_fft: int = 400,
                 hop_length: int = 160) -> np.ndarray:
    """
    简化版 MFCC 特征提取
    
    参数:
        audio: 音频信号, 形状 (n_samples,)
        sample_rate: 采样率
        n_mfcc: MFCC 系数数量
        n_fft: FFT 窗口大小
        hop_length: 跳跃长度
        
    返回:
        mfcc: MFCC 特征, 形状 (n_frames, n_mfcc)
    """
    n_frames = (len(audio) - n_fft) // hop_length + 1
    
    # 预加重
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    
    # 分帧
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        start = i * hop_length
        frames[i] = emphasized[start:start + n_fft]
    
    # 加窗
    window = np.hanning(n_fft)
    frames = frames * window
    
    # FFT
    fft_result = np.fft.rfft(frames, n=n_fft)
    power_spectrum = np.abs(fft_result) ** 2  # (n_frames, n_freqs)
    
    # 梅尔滤波器组 (简化)
    n_mels = 26
    n_freqs = n_fft // 2 + 1
    mel_filterbank = _mel_filterbank(n_mels, n_freqs, sample_rate)  # (n_mels, n_freqs)
    
    # 应用滤波器组: (n_frames, n_freqs) @ (n_freqs, n_mels) -> (n_frames, n_mels)
    mel_spectrum = np.dot(power_spectrum, mel_filterbank.T)
    mel_spectrum = np.where(mel_spectrum > 1e-10, mel_spectrum, 1e-10)
    log_mel = np.log(mel_spectrum + 1e-10)
    
    # DCT 得到 MFCC (简化)
    mfcc = np.zeros((n_frames, n_mfcc))
    for k in range(n_mfcc):
        for n in range(n_mels):
            mfcc[:, k] += log_mel[:, n] * np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    
    # 归一化
    mfcc = (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)
    
    return mfcc


def _mel_filterbank(n_mels: int, n_freqs: int, sample_rate: int) -> np.ndarray:
    """计算梅尔滤波器组"""
    # 频率转梅尔
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    # 梅尔频率范围
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # 对应到 FFT bins
    bin_points = np.floor((n_freqs) * hz_points / (sample_rate / 2)).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)
    
    # 构建滤波器 (n_mels, n_freqs)
    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            if bin_points[i + 1] != bin_points[i]:
                filterbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            if bin_points[i + 2] != bin_points[i + 1]:
                filterbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
    
    return filterbank




# =============================================================================
#                           CTC 损失和解码
# =============================================================================
class CTCDecoder:
    """
    CTC (Connectionist Temporal Classification) 解码器
    
    将模型输出转换为文本序列
    """
    
    def __init__(self, blank: int = 0):
        self.blank = blank
    
    def greedy_decode(self, logits: np.ndarray) -> List[int]:
        """
        贪婪解码
        
        参数:
            logits: 模型输出, 形状 (time_steps, vocab_size)
            
        返回:
            解码后的序列
        """
        predictions = np.argmax(logits, axis=-1)
        
        # 移除连续重复和空白
        decoded = []
        prev_token = None
        
        for token in predictions:
            if token != self.blank and token != prev_token:
                decoded.append(int(token))
            prev_token = token
        
        return decoded
    
    def beam_search_decode(self, logits: np.ndarray, beam_width: int = 10) -> List[int]:
        """
        束搜索解码
        
        参数:
            logits: 模型输出, 形状 (time_steps, vocab_size)
            beam_width: 束宽
        """
        # 计算概率
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        
        # 初始化
        beams = [{'seq': [], 'score': 0.0, 'last_char': self.blank}]
        
        for t in range(len(logits)):
            new_beams = []
            
            for beam in beams:
                for c in range(len(logits[t])):
                    prob = logits[t, c]
                    new_seq = beam['seq'].copy()
                    new_last = c
                    
                    # 处理重复
                    if c != self.blank and c != beam['last_char']:
                        new_seq.append(int(c))
                    
                    # 更新分数
                    new_score = beam['score'] + np.log(prob + 1e-10)
                    
                    new_beams.append({
                        'seq': new_seq,
                        'score': new_score,
                        'last_char': new_last
                    })
            
            # 保留 top-k
            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:beam_width]
        
        # 返回最佳路径
        best = max(beams, key=lambda x: x['score'])
        return best['seq']


# =============================================================================
#                           简化 ASR 模型
# =============================================================================
class SimpleASR:
    """
    简化 ASR 模型 (基于 RNN)
    
    使用 NumPy 实现，用于学习和理解 ASR 原理
    """
    
    def __init__(self, input_dim: int = 13, hidden_dim: int = 128,
                 vocab_size: int = 28, num_layers: int = 2):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            vocab_size: 词汇表大小
            num_layers: RNN 层数
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        np.random.seed(42)
        
        # 输入到隐藏层
        self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.1
        
        # 隐藏层到隐藏层 (循环)
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        
        # 隐藏层到输出
        self.W_hy = np.random.randn(hidden_dim, vocab_size) * 0.1
        
        # 偏置
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(vocab_size)
        
        print(f"\n{'='*60}")
        print("简化 ASR 模型初始化")
        print(f"{'='*60}")
        print(f"输入维度: {input_dim}")
        print(f"隐藏层维度: {hidden_dim}")
        print(f"词汇表大小: {vocab_size}")
        print(f"RNN 层数: {num_layers}")
        self._print_params()
    
    def _print_params(self):
        params = (self.W_xh.size + self.W_hh.size + self.W_hy.size +
                 self.b_h.size + self.b_y.size)
        print(f"总参数量: {params:,}")
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid 激活"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh 激活"""
        return np.tanh(x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单步前向传播
        
        参数:
            x_t: t 时刻输入 (batch, input_dim)
            h_prev: t-1 时刻隐藏状态 (batch, hidden_dim)
            
        返回:
            y_t: t 时刻输出 (batch, vocab_size)
            h_t: t 时刻隐藏状态 (batch, hidden_dim)
        """
        # 隐藏状态
        h_t = self._tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
        
        # 输出
        y_t = self._softmax(h_t @ self.W_hy + self.b_y)
        
        return y_t, h_t
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        前向传播
        
        参数:
            X: 输入序列 (batch, time_steps, input_dim)
            
        返回:
            outputs: 输出序列
            hidden_states: 隐藏状态列表
        """
        batch_size, time_steps, _ = X.shape
        
        outputs = np.zeros((batch_size, time_steps, self.vocab_size))
        hidden_states = [np.zeros((batch_size, self.hidden_dim))]
        
        h_prev = hidden_states[0]
        
        for t in range(time_steps):
            y_t, h_t = self.forward_step(X[:, t], h_prev)
            outputs[:, t] = y_t
            hidden_states.append(h_t)
            h_prev = h_t
        
        return outputs, hidden_states
    
    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 y_true: np.ndarray, hidden_states: List,
                 learning_rate: float = 0.01) -> float:
        """
        反向传播 (简化版，CTC 损失需要特殊处理)
        
        参数:
            X: 输入序列
            outputs: 模型输出
            y_true: 真实标签 (ctc 格式)
            hidden_states: 隐藏状态
            learning_rate: 学习率
        """
        batch_size, time_steps, _ = outputs.shape
        
        # 简化损失 (交叉熵)
        loss = 0
        for t in range(time_steps):
            for b in range(batch_size):
                loss -= np.log(outputs[b, t, y_true[b, t]] + 1e-10)
        loss /= (batch_size * time_steps)
        
        # 输出层梯度
        d_y = outputs.copy()
        for b in range(batch_size):
            for t in range(time_steps):
                d_y[b, t, y_true[b, t]] -= 1
        
        d_y /= (batch_size * time_steps)
        
        # 梯度累积
        d_W_hy = np.zeros_like(self.W_hy)
        d_b_y = np.zeros_like(self.b_y)
        d_h = np.zeros((batch_size, self.hidden_dim))
        
        for t in reversed(range(time_steps)):
            # 隐藏层梯度
            h_prev = hidden_states[t]
            d_h_next = d_h @ self.W_hh.T
            
            # tanh 导数
            d_tanh = (1 - h_prev ** 2)
            
            # 累计梯度
            d_W_hy += hidden_states[t].T @ d_y[:, t]
            d_b_y += np.sum(d_y[:, t], axis=0)
            
            # 隐藏层梯度
            d_h = d_y[:, t] @ self.W_hy.T * d_tanh + d_h_next * d_tanh
        
        d_W_hy /= batch_size
        d_b_y /= batch_size
        
        # 更新权重
        self.W_hy -= learning_rate * d_W_hy
        self.b_y -= learning_rate * d_b_y
        
        return loss
    
    def predict(self, X: np.ndarray, blank: int = 0) -> List[List[int]]:
        """
        预测
        
        参数:
            X: 输入序列
            blank: 空白符号索引
            
        返回:
            解码后的序列
        """
        outputs, _ = self.forward(X)
        
        decoder = CTCDecoder(blank=blank)
        predictions = []
        
        for b in range(outputs.shape[0]):
            decoded = decoder.greedy_decode(outputs[b])
            predictions.append(decoded)
        
        return predictions


# =============================================================================
#                           示例
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("ASR 模型示例 - NumPy 实现")
    print("="*60)
    
    np.random.seed(42)
    
    # ========================================
    # 示例 1: MFCC 特征提取
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: MFCC 特征提取")
    print("-"*40)
    
    # 模拟音频信号 (1秒)
    sample_rate = 16000
    duration = 1
    t = np.linspace(0, duration, sample_rate)
    
    # 合成简单音频 (440Hz + 880Hz)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    audio += np.random.randn(len(audio)) * 0.1  # 添加噪声
    
    # 提取 MFCC
    mfcc = extract_mfcc(audio, sample_rate=sample_rate, n_mfcc=13)
    print(f"音频长度: {len(audio)} samples")
    print(f"MFCC 特征形状: {mfcc.shape}")
    print(f"MFCC 特征范围: [{mfcc.min():.4f}, {mfcc.max():.4f}]")
    
    # ========================================
    # 示例 2: CTC 解码
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: CTC 解码")
    print("-"*40)
    
    # 模拟模型输出
    time_steps = 50
    vocab_size = 28  # 26字母 + 空白 + 未知
    
    logits = np.random.randn(time_steps, vocab_size)
    logits = logits / np.sum(np.exp(logits), axis=-1, keepdims=True)  # softmax
    
    decoder = CTCDecoder(blank=0)
    
    # 贪婪解码
    decoded_greedy = decoder.greedy_decode(logits)
    print(f"贪婪解码结果: {decoded_greedy[:10]}...")
    print(f"解码长度: {len(decoded_greedy)}")
    
    # 束搜索解码
    decoded_beam = decoder.beam_search_decode(logits, beam_width=5)
    print(f"束搜索解码结果: {decoded_beam[:10]}...")
    print(f"解码长度: {len(decoded_beam)}")
    
    # ========================================
    # 示例 3: 简化 ASR 模型
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: 简化 ASR 模型训练")
    print("-"*40)
    
    # 模拟数据
    n_samples = 50
    time_steps = 100
    input_dim = 13
    vocab_size = 28
    
    X = np.random.randn(n_samples, time_steps, input_dim)
    y = np.random.randint(1, vocab_size, (n_samples, time_steps))  # 1 = 空白
    
    # 划分
    split = 40
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 创建模型
    model = SimpleASR(input_dim=input_dim, hidden_dim=64, vocab_size=vocab_size)
    
    # 训练
    print("\n开始训练...")
    for epoch in range(5):
        outputs, hidden_states = model.forward(X_train)
        loss = model.backward(X_train, outputs, y_train, hidden_states, learning_rate=0.1)
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/5] - Loss: {loss:.4f}")
    
    # 预测
    print("\n预测示例:")
    predictions = model.predict(X_test[:3], blank=0)
    for i, pred in enumerate(predictions):
        print(f"  样本 {i+1}: {pred[:10]}...")
    
    print("\n" + "="*60)
    print("ASR 模型示例完成!")
    print("="*60)
