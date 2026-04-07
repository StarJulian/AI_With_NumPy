"""
================================================================================
                        NLP 模型 - NumPy 实现
================================================================================
包含:
    - TextCNN: 文本分类
    - Word2Vec: 词向量训练 (Skip-Gram)
    - N-Gram: 语言模型
    - HMM: 隐马尔可夫模型
================================================================================
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import re
from collections import Counter


# =============================================================================
#                           文本处理工具
# =============================================================================
class Tokenizer:
    """简单分词器"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_counts = Counter()
    
    def fit(self, texts: List[str]):
        """构建词汇表"""
        # 分词
        for text in texts:
            tokens = self._tokenize(text)
            self.word_counts.update(tokens)
        
        # 取最常见的词
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        for word, count in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"词汇表大小: {len(self.word2idx)}")
        return self
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def texts_to_sequences(self, texts: List[str], max_len: Optional[int] = None) -> np.ndarray:
        """文本转序列"""
        sequences = []
        for text in texts:
            tokens = self._tokenize(text)
            seq = [self.word2idx.get(w, 1) for w in tokens]  # 1 = <UNK>
            
            if max_len:
                if len(seq) > max_len:
                    seq = seq[:max_len]
                else:
                    seq = seq + [0] * (max_len - len(seq))  # 0 = <PAD>
            
            sequences.append(seq)
        
        return np.array(sequences)
    
    def decode(self, sequence: List[int]) -> str:
        """序列转文本"""
        words = []
        for idx in sequence:
            if idx == 0:  # PAD
                continue
            if idx == 3:  # EOS
                break
            words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)


# =============================================================================
#                           TextCNN 实现 (NumPy)
# =============================================================================
class TextCNNNumPy:
    """
    TextCNN: 文本分类 NumPy 实现
    
    结构:
        Embedding -> Conv1D (多尺度) -> MaxPool -> Concat -> FC
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128,
                 num_classes: int = 2, filter_sizes: List[int] = None,
                 num_filters: int = 100):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        np.random.seed(42)
        
        # 嵌入层
        self.embedding = np.random.randn(self.vocab_size, self.embed_dim) * 0.1
        
        # 卷积层
        self.convs = []
        for fs in self.filter_sizes:
            W = np.random.randn(fs, self.embed_dim, self.num_filters) * 0.1
            b = np.zeros(self.num_filters)
            self.convs.append((W, b))
        
        # 全连接层
        fc_input_dim = len(self.filter_sizes) * self.num_filters
        self.W_fc = np.random.randn(fc_input_dim, self.num_classes) * 0.1
        self.b_fc = np.zeros(self.num_classes)
        
        print(f"\n{'='*60}")
        print("TextCNN 初始化")
        print(f"{'='*60}")
        print(f"词汇表: {self.vocab_size}, 嵌入维度: {self.embed_dim}")
        print(f"卷积核尺寸: {self.filter_sizes}, 数量: {self.num_filters}")
        self._print_params()
    
    def _print_params(self):
        """打印参数量"""
        emb_params = self.embedding.size
        conv_params = sum(w.size + b.size for w, b in self.convs)
        fc_params = self.W_fc.size + self.b_fc.size
        total = emb_params + conv_params + fc_params
        
        print(f"\n{'层名称':<15} {'参数量':<15}")
        print("-" * 30)
        print(f"{'Embedding':<15} {emb_params:<15,}")
        print(f"{'Conv1D':<15} {conv_params:<15,}")
        print(f"{'FC':<15} {fc_params:<15,}")
        print("-" * 30)
        print(f"{'总计':<15} {total:<15,}")
    
    def _conv1d(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        1D 卷积
        
        参数:
            x: (seq_len, embed_dim)
            W: (filter_size, embed_dim, num_filters)
        """
        seq_len, embed_dim = x.shape
        filter_size, _, num_filters = W.shape
        
        out_len = seq_len - filter_size + 1
        output = np.zeros((out_len, num_filters))
        
        for i in range(out_len):
            for f in range(num_filters):
                for fs in range(filter_size):
                    output[i, f] += np.sum(x[i+fs] * W[fs, :, f])
        
        output = output + b  # 广播
        output = np.maximum(0, output)  # ReLU
        
        return output
    
    def forward(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List]:
        """
        前向传播
        
        参数:
            x: 输入 token IDs, 形状 (batch, seq_len)
            
        返回:
            output: 分类 logits
            cache: 缓存
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # 嵌入
        embedded = np.zeros((batch_size, seq_len, self.embed_dim))
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = x[b, s]
                if 0 <= token_id < self.vocab_size:
                    embedded[b, s] = self.embedding[token_id]
        
        cache = [embedded]
        
        # 多尺度卷积
        conv_outputs = []
        for (W, b) in self.convs:
            pooled = np.zeros((batch_size, self.num_filters))
            
            for b in range(batch_size):
                conv_out = self._conv1d(embedded[b], W, b)
                pooled[b] = np.max(conv_out, axis=0)  # MaxPool
            
            conv_outputs.append(pooled)
        
        # 拼接
        concat = np.concatenate(conv_outputs, axis=1)
        cache.append(concat)
        
        # 全连接
        output = concat @ self.W_fc + self.b_fc
        
        # Softmax
        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
        probs = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        
        cache.append(probs)
        
        return probs, cache
    
    def backward(self, y_true: np.ndarray, output: np.ndarray, cache: List,
                 learning_rate: float = 0.01) -> float:
        """反向传播"""
        batch_size = y_true.shape[0]
        
        # 损失
        loss = -np.mean(np.sum(y_true * np.log(output + 1e-8), axis=1))
        
        # 输出层梯度
        d_output = output - y_true
        
        # FC 梯度
        concat = cache[1]
        d_W_fc = concat.T @ d_output / batch_size
        d_b_fc = np.mean(d_output, axis=0)
        
        # 更新 FC
        self.W_fc -= learning_rate * d_W_fc
        self.b_fc -= learning_rate * d_b_fc
        
        return loss
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32,
            learning_rate: float = 0.01, verbose: bool = True) -> Dict:
        """训练模型"""
        n_samples = X_train.shape[0]
        
        # One-hot 编码
        y_onehot = np.eye(self.num_classes)[y_train]
        
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_onehot[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output, cache = self.forward(X_batch)
                loss = self.backward(y_batch, output, cache, learning_rate)
                
                epoch_loss += loss
                n_batches += 1
            
            epoch_loss /= n_batches
            history['train_loss'].append(epoch_loss)
            
            # 准确率
            predictions = self.predict(X_train)
            acc = np.mean(predictions == y_train)
            history['train_acc'].append(acc)
            
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {acc:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        output, _ = self.forward(X, training=False)
        return np.argmax(output, axis=1)


# =============================================================================
#                           Word2Vec 实现 (NumPy)
# =============================================================================
class Word2VecNumPy:
    """
    Word2Vec Skip-Gram 模型 NumPy 实现
    
    使用负采样优化
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 window_size: int = 5, learning_rate: float = 0.025,
                 num_negative: int = 5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.num_negative = num_negative
        
        np.random.seed(42)
        
        # 初始化嵌入
        self.center_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.context_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # 词汇统计
        self.word_counts = Counter()
        
        print(f"\n{'='*60}")
        print("Word2Vec 初始化")
        print(f"{'='*60}")
        print(f"词汇表: {vocab_size}, 嵌入维度: {embedding_dim}")
        print(f"窗口大小: {window_size}, 负采样数: {num_negative}")
    
    def build_vocab(self, texts: List[str]):
        """构建词汇表"""
        for text in texts:
            tokens = text.lower().split()
            self.word_counts.update(tokens)
        
        # 创建词到索引的映射
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        
        for word, _ in self.word_counts.most_common(self.vocab_size - 1):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"实际词汇表大小: {self.vocab_size}")
        
        # 初始化嵌入矩阵
        self.center_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        self.context_embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
    
    def _get_training_pairs(self, text: str) -> List[Tuple[int, int]]:
        """获取训练对 (center, context)"""
        tokens = text.lower().split()
        indices = [self.word2idx.get(w, 0) for w in tokens]
        
        pairs = []
        for i, center in enumerate(indices):
            start = max(0, i - self.window_size)
            end = min(len(indices), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    pairs.append((center, indices[j]))
        
        return pairs
    
    def _negative_sampling(self, positive_context: int, num_samples: int) -> np.ndarray:
        """负采样"""
        return np.random.randint(0, self.vocab_size, num_samples)
    
    def train_step(self, center: np.ndarray, context: np.ndarray,
                   negative: np.ndarray) -> float:
        """单步训练"""
        batch_size = len(center)
        
        # 获取嵌入
        center_emb = self.center_embeddings[center]  # (batch, embed_dim)
        context_emb = self.context_embeddings[context]  # (batch, embed_dim)
        neg_emb = self.context_embeddings[negative]  # (batch, num_neg, embed_dim)
        
        # 正样本分数
        pos_score = np.sum(center_emb * context_emb, axis=1)
        pos_loss = -np.mean(np.log(1 / (1 + np.exp(-pos_score)) + 1e-10))
        
        # 负样本分数
        neg_score = np.sum(center_emb[:, np.newaxis, :] * neg_emb, axis=2)
        neg_loss = -np.mean(np.sum(np.log(1 / (1 + np.exp(-neg_score)) + 1e-10), axis=1))
        
        loss = pos_loss + neg_loss
        
        # 梯度
        pos_grad = (center_emb - context_emb) / batch_size
        context_grad = (context_emb - center_emb) / batch_size
        
        # 更新嵌入
        for i, c in enumerate(center):
            self.center_embeddings[c] -= self.learning_rate * pos_grad[i]
        
        for i, ctx in enumerate(context):
            self.context_embeddings[ctx] -= self.learning_rate * context_grad[i]
        
        return loss
    
    def fit(self, texts: List[str], epochs: int = 5, verbose: bool = True) -> Dict:
        """训练模型"""
        self.build_vocab(texts)
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            total_loss = 0
            n_pairs = 0
            
            for text in texts:
                pairs = self._get_training_pairs(text)
                
                # 批次处理
                for i in range(0, len(pairs), 100):
                    batch = pairs[i:i+100]
                    
                    centers = np.array([p[0] for p in batch])
                    contexts = np.array([p[1] for p in batch])
                    negatives = np.array([
                        self._negative_sampling(ctx, self.num_negative)
                        for ctx in contexts
                    ])
                    
                    loss = self.train_step(centers, contexts, negatives)
                    total_loss += loss
                    n_pairs += 1
            
            avg_loss = total_loss / max(n_pairs, 1)
            history['loss'].append(avg_loss)
            
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
        return history
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """获取词嵌入"""
        idx = self.word2idx.get(word)
        if idx is not None:
            return self.center_embeddings[idx]
        return None
    
    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """查找最相似的词"""
        word_emb = self.get_embedding(word)
        if word_emb is None:
            return []
        
        similarities = []
        for idx, emb in enumerate(self.center_embeddings):
            if idx != 0:  # 跳过 <UNK>
                sim = np.dot(word_emb, emb) / (np.linalg.norm(word_emb) * np.linalg.norm(emb) + 1e-10)
                similarities.append((self.idx2word[idx], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# =============================================================================
#                           N-Gram 语言模型 (NumPy)
# =============================================================================
class NGramLM:
    """
    N-Gram 语言模型
    
    使用简单的最大似然估计 + 平滑
    """
    
    def __init__(self, n: int = 3, smoothing: float = 1e-5):
        """
        参数:
            n: N-gram 大小 (2 = bigram, 3 = trigram, etc.)
            smoothing: 平滑参数
        """
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set(['<S>', '</S>'])  # 起始和结束符
        
        print(f"\n{'='*60}")
        print(f"N-Gram 语言模型初始化 (n={n})")
        print(f"{'='*60}")
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        tokens = text.lower().split()
        return ['<S>'] * (self.n - 1) + tokens + ['</S>']
    
    def fit(self, texts: List[str]):
        """训练模型"""
        print("构建 N-gram 统计...")
        
        for text in texts:
            tokens = self._tokenize(text)
            self.vocab.update(tokens)
            
            # 统计 N-gram
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = ngram[:-1]
                
                self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1
        
        vocab_size = len(self.vocab)
        print(f"词汇表大小: {vocab_size}")
        print(f"N-gram 类型数: {len(self.ngram_counts)}")
        print(f"上下文类型数: {len(self.context_counts)}")
    
    def _get_prob(self, ngram: Tuple[str, ...]) -> float:
        """计算 N-gram 概率"""
        context = ngram[:-1]
        ngram_str = ngram
        
        ngram_count = self.ngram_counts.get(ngram_str, 0)
        context_count = self.context_counts.get(context, 0)
        
        vocab_size = len(self.vocab)
        
        # 加法平滑
        prob = (ngram_count + self.smoothing) / (context_count + self.smoothing * vocab_size)
        
        return prob
    
    def score(self, text: str) -> float:
        """计算文本的 log 概率"""
        tokens = self._tokenize(text)
        
        log_prob = 0.0
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            prob = self._get_prob(ngram)
            log_prob += np.log(prob + 1e-10)
        
        return log_prob
    
    def generate(self, seed: str = None, max_len: int = 20) -> str:
        """生成文本"""
        if seed:
            tokens = self._tokenize(seed)
            tokens = tokens[-(self.n-1):] if len(tokens) >= self.n-1 else ['<S>'] * (self.n - len(tokens)) + tokens
        else:
            tokens = ['<S>'] * (self.n - 1)
        
        generated = []
        
        for _ in range(max_len):
            context = tuple(tokens[-(self.n-1):] if len(tokens) >= self.n-1 else ['<S>'] * (self.n-1-len(tokens)) + tokens)
            
            # 找到所有可能的下一个词
            candidates = {}
            for ngram, count in self.ngram_counts.items():
                if ngram[:-1] == context:
                    candidates[ngram[-1]] = count
            
            if not candidates:
                break
            
            # 选择概率最高的词
            vocab_size = len(self.vocab)
            probs = np.array([candidates.get(w, self.smoothing) for w in self.vocab])
            probs = probs / probs.sum()
            
            idx = np.random.choice(len(self.vocab), p=probs)
            word = list(self.vocab)[idx]
            
            if word == '</S>':
                break
            
            generated.append(word)
            tokens.append(word)
        
        return ' '.join(generated)


# =============================================================================
#                           示例
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("NLP 模型示例 - NumPy 实现")
    print("="*60)
    
    np.random.seed(42)
    
    # ========================================
    # 示例 1: TextCNN
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: TextCNN 文本分类")
    print("-"*40)
    
    # 模拟文本数据
    n_samples = 500
    vocab_size = 5000
    seq_len = 50
    
    X = np.random.randint(1, vocab_size, (n_samples, seq_len))
    y = np.random.randint(0, 2, n_samples)
    
    # 划分
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 训练
    model = TextCNNNumPy(vocab_size=vocab_size, embed_dim=64, num_classes=2,
                         filter_sizes=[2, 3, 4], num_filters=50)
    
    print("\n开始训练...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, learning_rate=0.01)
    
    # 测试
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    # ========================================
    # 示例 2: Word2Vec
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: Word2Vec 词向量")
    print("-"*40)
    
    # 模拟文本
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with many layers",
        "natural language processing deals with text data",
        "computer vision processes and analyzes images and videos",
        "reinforcement learning learns from rewards and punishments",
        "supervised learning uses labeled training data",
        "unsupervised learning finds patterns in unlabeled data",
    ]
    
    w2v = Word2VecNumPy(vocab_size=1000, embedding_dim=50, window_size=2)
    w2v.fit(sample_texts, epochs=10)
    
    # 测试相似词
    test_word = "learning"
    similar = w2v.most_similar(test_word, top_k=3)
    print(f"\n与 '{test_word}' 最相似的词:")
    for word, sim in similar:
        print(f"  {word}: {sim:.4f}")
    
    # ========================================
    # 示例 3: N-Gram 语言模型
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: N-Gram 语言模型")
    print("-"*40)
    
    training_texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a cat is a good pet",
        "dogs are loyal animals",
        "the cat and the dog played",
        "a pet can be a cat or a dog",
    ]
    
    # Trigram 模型
    ngram = NGramLM(n=3)
    ngram.fit(training_texts)
    
    # 计算句子概率
    test_sentences = [
        "the cat sat on the mat",
        "a dog is a good pet",
    ]
    
    print("\n句子概率:")
    for sent in test_sentences:
        log_prob = ngram.score(sent)
        print(f"  '{sent}'")
        print(f"    Log Prob: {log_prob:.4f}, Perplexity: {np.exp(-log_prob/len(sent.split())):.4f}")
    
    # 生成文本
    print("\n生成的文本:")
    for _ in range(3):
        generated = ngram.generate(seed="the", max_len=10)
        print(f"  '{generated}'")
    
    print("\n" + "="*60)
    print("NLP 模型示例完成!")
    print("="*60)
