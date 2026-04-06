"""
================================================================================
                    PyTorch NLP 模块
================================================================================
自然语言处理算法实现
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional


class TextClassifier(nn.Module):
    """文本分类器 (基于 CNN)"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 num_classes: int = 2, num_filters: int = 100,
                 filter_sizes: List[int] = [3, 4, 5], dropout: float = 0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * num_filters, num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = torch.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outputs.append(c)
        
        x = torch.cat(conv_outputs, dim=1)
        return self.fc(x)


class Word2Vec(nn.Module):
    """Word2Vec (Skip-gram) 模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化
        nn.init.uniform_(self.target_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.zeros_(self.context_embeddings.weight)
    
    def forward(self, target: torch.Tensor, context: torch.Tensor, 
                neg_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Skip-gram 前向传播
        
        参数:
            target: 目标词索引 (batch,)
            context: 上下文词索引 (batch,)
            neg_samples: 负采样词索引 (batch, n_negatives)
        """
        # 正样本
        target_emb = self.target_embeddings(target)  # (batch, embed_dim)
        context_emb = self.context_embeddings(context)  # (batch, embed_dim)
        pos_score = torch.sum(target_emb * context_emb, dim=1)
        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score)
        )
        
        # 负样本
        neg_emb = self.context_embeddings(neg_samples)  # (batch, n_neg, embed_dim)
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(2)
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )
        
        return pos_loss, neg_loss


class SequenceTagger(nn.Module):
    """序列标注器 (用于 POS 标注、NER 等)"""
    
    def __init__(self, vocab_size: int, tag_size: int, 
                 embed_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)
        
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        
        return self.fc(lstm_out)  # (batch, seq_len, tag_size)


class Attention(nn.Module):
    """注意力机制"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch, seq_len, hidden_dim)
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 加权求和
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_dim)
        
        return context, attention_weights


class Seq2Seq(nn.Module):
    """Seq2Seq 翻译模型"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 embed_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                                    dropout=dropout if num_layers > 1 else 0,
                                    batch_first=True, bidirectional=True)
        
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim * 2, num_layers,
                                    dropout=dropout if num_layers > 1 else 0,
                                    batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 2, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # 编码
        src_emb = self.dropout(self.encoder_embedding(src))
        _, (hidden, cell) = self.encoder_lstm(src_emb)
        
        # 初始化解码器隐藏状态
        hidden = torch.cat([hidden[::2], hidden[1::2]], dim=2)
        cell = torch.cat([cell[::2], cell[1::2]], dim=2)
        
        # 解码
        outputs = []
        decoder_input = tgt[:, 0]  # <sos>
        
        for t in range(1, tgt_len):
            decoder_emb = self.dropout(self.decoder_embedding(decoder_input))
            
            lstm_out, (hidden, cell) = self.decoder_lstm(decoder_emb.unsqueeze(1), (hidden, cell))
            
            prediction = self.fc(lstm_out.squeeze(1))
            outputs.append(prediction)
            
            # Teacher forcing
            decoder_input = tgt[:, t] if np.random.random() < teacher_forcing_ratio else prediction.argmax(1)
        
        return torch.stack(outputs, dim=1)


class TransformerNLP(nn.Module):
    """基于 Transformer 的 NLP 模型"""
    
    def __init__(self, vocab_size: int, num_classes: int,
                 d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 平均池化
        return self.fc(x)


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


if __name__ == "__main__":
    print("="*60)
    print("PyTorch NLP 模块示例")
    print("="*60)
    
    # 文本分类示例
    print("\n文本分类器:")
    vocab_size = 10000
    batch_size = 16
    seq_len = 50
    
    texts = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,))
    
    model = TextClassifier(vocab_size=vocab_size, embed_dim=128, num_classes=2)
    outputs = model(texts)
    print(f"输入形状: {texts.shape}")
    print(f"输出形状: {outputs.shape}")
    
    # 序列标注示例
    print("\n序列标注器:")
    seq_len = 20
    tags = torch.randint(0, 10, (batch_size, seq_len))
    
    tagger = SequenceTagger(vocab_size=vocab_size, tag_size=10)
    tag_outputs = tagger(tags)
    print(f"输入形状: {tags.shape}")
    print(f"输出形状: {tag_outputs.shape}")
    
    print("\n" + "="*60)
