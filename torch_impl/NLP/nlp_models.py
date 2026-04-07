"""
================================================================================
                        NLP 模型 - PyTorch 实现
================================================================================
包含:
    - TextCNN: 文本分类
    - Word2Vec: 词向量训练
    - Seq2Seq: 序列到序列模型
    - Transformer: 注意力机制
    - BERT: 预训练语言模型
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


# =============================================================================
#                           TextCNN 实现
# =============================================================================
class TextCNN(nn.Module):
    """
    TextCNN: Convolutional Neural Networks for Sentence Classification (2014)
    
    结构:
        Embedding -> Conv1D (多尺度) -> MaxPool -> Concat -> FC
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128,
                 num_classes: int = 2, filter_sizes: List[int] = None,
                 num_filters: int = 100, dropout: float = 0.5):
        super(TextCNN, self).__init__()
        
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入 token IDs, 形状 (batch, seq_len)
            
        返回:
            分类 logits, 形状 (batch, num_classes)
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embedding(x)
        
        # Transpose: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        conv_outputs = []
        for conv in self.convs:
            c = conv(x)  # (batch, num_filters, seq_len - filter_size + 1)
            c = F.relu(c)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outputs.append(c)
        
        # 拼接所有卷积结果
        x = torch.cat(conv_outputs, dim=1)  # (batch, len(filter_sizes) * num_filters)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def summary(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"\nTextCNN: vocab={self.embedding.num_embeddings}, embed={self.embedding.embedding_dim}")
        print(f"Filters: {self.filter_sizes}, num_filters={self.num_filters}")
        print(f"Parameters: {params:,}")


# =============================================================================
#                           Word2Vec 实现
# =============================================================================
class Word2VecSkipGram(nn.Module):
    """
    Word2Vec Skip-Gram 模型
    
    目标: 最大化 P(w_context | w_center)
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        super(Word2VecSkipGram, self).__init__()
        
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.uniform_(self.center_embeddings.weight, -0.5/self.embedding_dim, 0.5/self.embedding_dim)
        nn.init.zeros_(self.center_embeddings.weight)
        nn.init.uniform_(self.context_embeddings.weight, -0.5/self.embedding_dim, 0.5/self.embedding_dim)
        nn.init.zeros_(self.context_embeddings.weight)
    
    def forward(self, center: torch.Tensor, context: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        参数:
            center: 中心词 IDs, 形状 (batch,)
            context: 上下文词 IDs, 形状 (batch,)
            negative: 负采样词 IDs, 形状 (batch, num_negative)
            
        返回:
            损失值
        """
        # 中心词和上下文词的嵌入
        center_emb = self.center_embeddings(center)  # (batch, embed_dim)
        context_emb = self.context_embeddings(context)  # (batch, embed_dim)
        neg_emb = self.context_embeddings(negative)  # (batch, num_neg, embed_dim)
        
        # 正样本损失
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # (batch,)
        pos_loss = F.logsigmoid(pos_score)
        
        # 负样本损失
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # (batch,)
        
        loss = -(pos_loss + neg_loss).mean()
        
        return loss
    
    def get_embedding(self, word_id: int) -> torch.Tensor:
        """获取词嵌入"""
        return self.center_embeddings.weight[word_id]


class Word2VecCBOW(nn.Module):
    """
    Word2Vec CBOW (Continuous Bag of Words) 模型
    
    目标: 根据上下文预测中心词
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 context_size: int = 2):
        super(Word2VecCBOW, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(context_size * 2 * embedding_dim, vocab_size)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        参数:
            context: 上下文词 IDs, 形状 (batch, context_size * 2)
            
        返回:
            预测 logits, 形状 (batch, vocab_size)
        """
        emb = self.embedding(context)  # (batch, context_size*2, embed_dim)
        emb = emb.view(emb.size(0), -1)  # (batch, context_size*2*embed_dim)
        out = self.linear(emb)
        return out


# =============================================================================
#                           Seq2Seq 实现
# =============================================================================
class Encoder(nn.Module):
    """Seq2Seq 编码器"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.2):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        参数:
            x: 输入序列, 形状 (batch, seq_len)
            
        返回:
            outputs: 所有时刻的输出, 形状 (batch, seq_len, hidden_dim*2)
            hidden: 最终隐藏状态
        """
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
        outputs, hidden = self.lstm(embedded)
        
        return outputs, hidden
    
    def summary(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"Encoder: vocab->embed->BiLSTM({self.hidden_dim})")
        print(f"Parameters: {params:,}")


class Attention(nn.Module):
    """Bahdanau 注意力机制"""
    
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            hidden: 解码器当前隐藏状态, 形状 (batch, hidden_dim)
            encoder_outputs: 编码器输出, 形状 (batch, seq_len, hidden_dim*2)
            mask: 掩码, 形状 (batch, seq_len)
            
        返回:
            context: 上下文向量, 形状 (batch, hidden_dim*2)
            attn_weights: 注意力权重, 形状 (batch, seq_len)
        """
        batch_size, seq_len = encoder_outputs.shape[0], encoder_outputs.shape[1]
        
        # 重复 hidden 以匹配 encoder_outputs 长度
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim*2)
        
        # 计算注意力分数
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch, seq_len)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        attn_weights = F.softmax(attention, dim=1)
        
        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights


class Seq2SeqDecoder(nn.Module):
    """Seq2Seq 解码器 with Attention"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.2):
        super(Seq2SeqDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim * 2)
        self.lstm = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim * 2,
                           num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 4, vocab_size)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, hidden: Tuple,
                encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple, torch.Tensor]:
        """
        参数:
            x: 输入 token, 形状 (batch, 1)
            hidden: 隐藏状态
            encoder_outputs: 编码器输出
            
        返回:
            prediction: 预测 logits
            hidden: 更新后的隐藏状态
            attn_weights: 注意力权重
        """
        # 获取解码器隐藏状态
        if isinstance(hidden, tuple):
            dec_hidden = (hidden[0][-1], hidden[1][-1])  # 取最后一层
        else:
            dec_hidden = hidden[-1]
        
        # 计算注意力
        context, attn_weights = self.attention(dec_hidden, encoder_outputs)
        
        # 嵌入
        embedded = self.dropout(self.embedding(x))
        
        # 拼接嵌入和上下文
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        
        # LSTM
        output, hidden = self.lstm(lstm_input, hidden)
        
        # 输出
        prediction = self.fc(torch.cat((output.squeeze(1), context), dim=1))
        
        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    """Seq2Seq 完整模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 1,
                 dropout: float = 0.2):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Seq2SeqDecoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        参数:
            src: 源序列, 形状 (batch, src_len)
            tgt: 目标序列, 形状 (batch, tgt_len)
            teacher_forcing_ratio: 教师强制比例
            
        返回:
            outputs: 预测序列, 形状 (batch, tgt_len, vocab_size)
        """
        batch_size, tgt_len = tgt.shape
        
        outputs = torch.zeros(batch_size, tgt_len, self.decoder.fc.out_features).to(tgt.device)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src)
        
        # 解码
        x = tgt[:, 0]  # 起始符
        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(x.unsqueeze(1), hidden, encoder_outputs)
            outputs[:, t] = output
            
            # 教师强制
            if torch.rand(1).item() < teacher_forcing_ratio:
                x = tgt[:, t]
            else:
                x = output.argmax(1)
        
        return outputs
    
    def summary(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"Seq2Seq: embed->BiLSTM({self.encoder.hidden_dim})")
        print(f"Parameters: {params:,}")


# =============================================================================
#                           Transformer 实现
# =============================================================================
class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer 文本分类模型
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, num_classes: int,
                 max_len: int = 512, dropout: float = 0.1):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            x: 输入序列, 形状 (batch, seq_len)
            mask: 注意力掩码
            
        返回:
            分类 logits, 形状 (batch, num_classes)
        """
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoding(embedded)
        
        encoded = self.transformer(embedded, src_key_padding_mask=mask)
        
        # 使用 [CLS] 或平均
        pooled = encoded[:, 0]  # [CLS] token
        
        output = self.dropout(pooled)
        output = self.fc(output)
        
        return output
    
    def summary(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"Transformer: vocab={self.embedding.num_embeddings}, embed={self.embedding.embedding_dim}")
        print(f"Parameters: {params:,}")


# =============================================================================
#                           BERT 实现 (简化版)
# =============================================================================
class BERTEmbedding(nn.Module):
    """BERT 嵌入层"""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 512,
                 dropout: float = 0.1):
        super(BERTEmbedding, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.segment_embedding = nn.Embedding(2, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
    
    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        参数:
            token_ids: token IDs, 形状 (batch, seq_len)
            segment_ids: 句子段落 ID, 形状 (batch, seq_len)
            
        返回:
            嵌入向量, 形状 (batch, seq_len, embed_dim)
        """
        seq_len = token_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(position_ids)
        seg_emb = self.segment_embedding(segment_ids)
        
        embeddings = token_emb + pos_emb + seg_emb
        
        return self.dropout(embeddings)


class BERTModel(nn.Module):
    """
    BERT (Bidirectional Encoder Representations from Transformers)
    
    简化版本，仅包含预训练任务部分
    """
    
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 768,
                 num_heads: int = 12, num_layers: int = 12,
                 intermediate_dim: int = 3072, num_classes: int = 2,
                 max_len: int = 512, dropout: float = 0.1):
        super(BERTModel, self).__init__()
        
        self.embedding = BERTEmbedding(vocab_size, embed_dim, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=intermediate_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 预训练任务头
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.nsp_head = nn.Linear(embed_dim, 2)
        
        # 分类任务头
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
    
    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                task: str = 'classify') -> torch.Tensor:
        """
        参数:
            token_ids: token IDs, 形状 (batch, seq_len)
            segment_ids: 句子段落 ID, 形状 (batch, seq_len)
            attention_mask: 注意力掩码
            task: 任务类型 ('mlm', 'nsp', 'classify')
            
        返回:
            任务输出
        """
        # 嵌入
        hidden = self.embedding(token_ids)
        
        # Transformer
        encoded = self.transformer(hidden, src_key_padding_mask=attention_mask)
        
        # 取 [CLS] token
        cls_output = encoded[:, 0]
        
        if task == 'mlm':
            return self.mlm_head(encoded)
        elif task == 'nsp':
            return self.nsp_head(cls_output)
        else:  # classify
            return self.classifier(cls_output)
    
    def summary(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"BERT: vocab={self.vocab_size}, embed={self.embed_dim}")
        print(f"Parameters: {params:,}")
        print(f"  - MLM Head: {self.mlm_head.weight.numel():,}")
        print(f"  - NSP Head: {self.nsp_head.weight.numel():,}")
        print(f"  - Classifier: {self.classifier.weight.numel():,}")


# =============================================================================
#                           示例
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("NLP 模型示例 - PyTorch 实现")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========================================
    # 示例 1: TextCNN
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: TextCNN")
    print("-"*40)
    
    textcnn = TextCNN(vocab_size=10000, embed_dim=128, num_classes=2)
    textcnn.to(device)
    textcnn.summary()
    
    x = torch.randint(0, 10000, (32, 50)).to(device)  # (batch, seq_len)
    output = textcnn(x)
    print(f"输入: {x.shape} -> 输出: {output.shape}")
    
    # ========================================
    # 示例 2: Word2Vec
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: Word2Vec Skip-Gram")
    print("-"*40)
    
    w2v = Word2VecSkipGram(vocab_size=10000, embedding_dim=128)
    w2v.to(device)
    
    center = torch.randint(1, 10000, (32,)).to(device)
    context = torch.randint(1, 10000, (32,)).to(device)
    negative = torch.randint(1, 10000, (32, 5)).to(device)
    
    loss = w2v(center, context, negative)
    print(f"输入: center={center.shape}, context={context.shape}, negative={negative.shape}")
    print(f"损失: {loss.item():.4f}")
    
    # ========================================
    # 示例 3: Seq2Seq
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: Seq2Seq")
    print("-"*40)
    
    seq2seq = Seq2Seq(vocab_size=10000, embed_dim=128, hidden_dim=256)
    seq2seq.to(device)
    seq2seq.summary()
    
    src = torch.randint(1, 10000, (16, 30)).to(device)  # (batch, src_len)
    tgt = torch.randint(1, 10000, (16, 25)).to(device)  # (batch, tgt_len)
    
    output = seq2seq(src, tgt, teacher_forcing_ratio=0.5)
    print(f"源序列: {src.shape} -> 目标序列: {tgt.shape}")
    print(f"输出: {output.shape}")
    
    # ========================================
    # 示例 4: Transformer
    # ========================================
    print("\n" + "-"*40)
    print("示例 4: Transformer 分类器")
    print("-"*40)
    
    transformer = TransformerClassifier(
        vocab_size=10000, embed_dim=256, num_heads=8,
        num_layers=6, hidden_dim=1024, num_classes=2
    )
    transformer.to(device)
    transformer.summary()
    
    x = torch.randint(0, 10000, (32, 100)).to(device)
    output = transformer(x)
    print(f"输入: {x.shape} -> 输出: {output.shape}")
    
    # ========================================
    # 示例 5: BERT
    # ========================================
    print("\n" + "-"*40)
    print("示例 5: BERT 预训练模型")
    print("-"*40)
    
    bert = BERTModel(vocab_size=30522, embed_dim=256, num_heads=4, num_layers=4)
    bert.to(device)
    bert.summary()
    
    token_ids = torch.randint(0, 30522, (4, 128)).to(device)
    segment_ids = torch.zeros(4, 128, dtype=torch.long).to(device)
    
    # MLM 任务
    mlm_output = bert(token_ids, segment_ids, task='mlm')
    print(f"MLM 输出: {mlm_output.shape}")
    
    # 分类任务
    clf_output = bert(token_ids, segment_ids, task='classify')
    print(f"分类输出: {clf_output.shape}")
    
    print("\n" + "="*60)
    print("NLP 模型示例完成!")
    print("="*60)
