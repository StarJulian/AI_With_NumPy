"""
================================================================================
                    PyTorch ASR 模块
================================================================================
自动语音识别算法实现
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class Wav2Vec2(nn.Module):
    """简化版 Wav2Vec 2.0 模型"""
    
    def __init__(self, num_features: int = 80, hidden_dim: int = 512,
                 num_layers: int = 12, num_heads: int = 8,
                 vocab_size: int = 32):
        super().__init__()
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(num_features, 512, kernel_size=10, stride=5, padding=3),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 投影层
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 量化器 (简化)
        self.quantizer = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            x: 音频特征, 形状 (batch, num_features, time_steps)
        
        返回:
            编码输出, 量化特征 (可选)
        """
        # 特征编码
        x = self.feature_encoder(x)  # (batch, 512, T')
        
        # 转换为序列格式
        x = x.permute(0, 2, 1)  # (batch, T', 512)
        
        # Transformer 编码
        x = self.transformer(x)
        
        # 投影
        x = self.projection(x)
        
        return x, None


class CTCModel(nn.Module):
    """CTC (Connectionist Temporal Classification) 模型"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256,
                 num_layers: int = 3, vocab_size: int = 32):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 音频特征, 形状 (batch, time_steps, input_dim)
        
        返回:
             logits, 形状 (batch, time_steps, vocab_size)
        """
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeepSpeech2(nn.Module):
    """Deep Speech 2 模型"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 800,
                 num_layers: int = 5, vocab_size: int = 32,
                 bidirectional: bool = True):
        super().__init__()
        
        self.bidirectional = bidirectional
        
        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        # RNN 层
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # 输出层
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 音频特征, 形状 (batch, time_steps, input_dim)
        
        返回:
             logits, 形状 (batch, time_steps, vocab_size)
        """
        # 卷积: (batch, time_steps, input_dim) -> (batch, input_dim, time_steps)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (batch, new_time_steps, hidden_dim)
        
        # RNN
        x, _ = self.rnn(x)
        
        # 全连接
        return self.fc(x)


class AttentionASR(nn.Module):
    """基于注意力机制的 ASR 模型"""
    
    def __init__(self, input_dim: int = 80, d_model: int = 256,
                 nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, vocab_size: int = 32,
                 max_len: int = 500):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 解码器
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)
        
        # 输出投影
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        参数:
            src: 音频特征, 形状 (batch, src_len, input_dim)
            tgt: 目标文本, 形状 (batch, tgt_len)
        
        返回:
            预测 logits, 形状 (batch, tgt_len, vocab_size)
        """
        # 编码
        src = self.input_proj(src)
        memory = self.encoder(src)
        
        # 解码
        tgt = self.decoder_embedding(tgt) * np.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        decoder_out = self.decoder(tgt)
        
        return self.fc(decoder_out)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def ctc_decode(logits: torch.Tensor, blank: int = 0,
               method: str = 'greedy') -> list:
    """
    CTC 解码
    
    参数:
        logits: 模型输出, 形状 (time_steps, vocab_size) 或 (batch, time_steps, vocab_size)
        blank: 空白符号索引
        method: 解码方法 ('greedy' 或 'beam_search')
    
    返回:
        解码后的文本序列列表
    """
    if len(logits.shape) == 3:
        # 批量处理
        results = []
        for i in range(logits.shape[0]):
            result = ctc_decode(logits[i], blank, method)
            results.append(result)
        return results
    
    # 贪婪解码
    predictions = logits.argmax(dim=-1).cpu().numpy()
    
    # 移除连续重复和空白
    decoded = []
    prev_token = None
    
    for token in predictions:
        if token != blank and token != prev_token:
            decoded.append(token)
        prev_token = token
    
    return decoded


if __name__ == "__main__":
    print("="*60)
    print("PyTorch ASR 模块示例")
    print("="*60)
    
    # CTC 模型
    print("\nCTC 模型:")
    ctc_model = CTCModel(input_dim=80, hidden_dim=256, vocab_size=32)
    x = torch.randn(2, 500, 80)  # batch, time_steps, features
    y = ctc_model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # Deep Speech 2
    print("\nDeep Speech 2:")
    ds2 = DeepSpeech2(input_dim=80, hidden_dim=256, vocab_size=32)
    y = ds2(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # CTC 解码示例
    print("\nCTC 解码示例:")
    logits = torch.randn(100, 32)
    decoded = ctc_decode(logits, blank=0)
    print(f"解码结果: {decoded[:20]}...")
    
    print("\n" + "="*60)
