"""
================================================================================
                        计算机视觉模型 - NumPy 实现
================================================================================
包含:
    - LeNet-5: 经典卷积神经网络 (简化版)
    - CNN 实现: 卷积、池化操作
    - 特征提取: HOG
================================================================================
"""

import numpy as np
from typing import Tuple, List, Optional


def convolution2d(image: np.ndarray, kernel: np.ndarray, 
                  stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    2D 卷积操作 (简化实现)
    
    参数:
        image: 输入图像 (H, W, C)
        kernel: 卷积核 (K, K, in_channels, out_channels)
        stride: 步长
        padding: 填充
        
    返回:
        卷积结果 (H', W', out_channels)
    """
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
        input_is_batch = False
    else:
        input_is_batch = True
    
    N, H, W, C_in = image.shape
    K, K, C_in_k, C_out = kernel.shape
    
    # 填充
    if padding > 0:
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        H, W = image.shape[1], image.shape[2]
    
    # 计算输出尺寸
    out_h = (H - K) // stride + 1
    out_w = (W - K) // stride + 1
    
    # 简化卷积实现
    output = np.zeros((N, out_h, out_w, C_out))
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            patch = image[:, h_start:h_start+K, w_start:w_start+K, :]  # (N, K, K, C_in)
            
            # 对每个输出通道计算卷积
            for c_out in range(C_out):
                for c_in in range(C_in):
                    output[:, i, j, c_out] += np.sum(
                        patch[:, :, :, c_in] * kernel[:, :, c_in, c_out], 
                        axis=(1, 2)
                    )
    
    if not input_is_batch:
        output = output[0]
    
    return output


def max_pooling2d(image: np.ndarray, pool_size: int = 2, 
                  stride: int = 2) -> np.ndarray:
    """2D 最大池化操作"""
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
        input_is_batch = False
    else:
        input_is_batch = True
    
    N, H, W, C = image.shape
    
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    output = np.zeros((N, out_h, out_w, C))
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            window = image[:, h_start:h_start+pool_size, w_start:w_start+pool_size, :]
            output[:, i, j, :] = np.max(window, axis=(1, 2))
    
    if not input_is_batch:
        output = output[0]
    
    return output


def avg_pooling2d(image: np.ndarray, pool_size: int = 2, 
                  stride: int = 2) -> np.ndarray:
    """2D 平均池化操作"""
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
        input_is_batch = False
    else:
        input_is_batch = True
    
    N, H, W, C = image.shape
    
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    output = np.zeros((N, out_h, out_w, C))
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            window = image[:, h_start:h_start+pool_size, w_start:w_start+pool_size, :]
            output[:, i, j, :] = np.mean(window, axis=(1, 2))
    
    if not input_is_batch:
        output = output[0]
    
    return output


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活函数"""
    return np.maximum(0, x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax 激活函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LeNet5NumPy:
    """
    LeNet-5: 经典卷积神经网络 NumPy 实现
    
    结构:
        Conv1(1->6, 5x5) -> AvgPool -> Conv2(6->16, 5x5) -> AvgPool 
        -> Flatten -> FC(400->120) -> FC(120->84) -> FC(84->10)
    """
    
    def __init__(self, input_size: int = 32, num_classes: int = 10):
        self.input_size = input_size
        self.num_classes = num_classes
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        np.random.seed(42)
        
        # Conv1: 1 -> 6 channels, 5x5
        self.W1 = np.random.randn(5, 5, 1, 6) * 0.1
        self.b1 = np.zeros(6)
        
        # Conv2: 6 -> 16 channels, 5x5
        self.W2 = np.random.randn(5, 5, 6, 16) * 0.1
        self.b2 = np.zeros(16)
        
        # 计算全连接层输入大小: 16 * 5 * 5 = 400
        fc_input_size = 400
        
        # FC1: 400 -> 120
        self.W3 = np.random.randn(fc_input_size, 120) * 0.1
        self.b3 = np.zeros(120)
        
        # FC2: 120 -> 84
        self.W4 = np.random.randn(120, 84) * 0.1
        self.b4 = np.zeros(84)
        
        # FC3: 84 -> num_classes
        self.W5 = np.random.randn(84, self.num_classes) * 0.1
        self.b5 = np.zeros(self.num_classes)
        
        print(f"\n{'='*60}")
        print("LeNet-5 初始化")
        print(f"{'='*60}")
        print(f"输入尺寸: {self.input_size}x{self.input_size}")
        print(f"类别数: {self.num_classes}")
        
        total_params = (self.W1.size + self.b1.size + self.W2.size + self.b2.size +
                       self.W3.size + self.b3.size + self.W4.size + self.b4.size +
                       self.W5.size + self.b5.size)
        print(f"总参数量: {total_params:,}")
    
    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List]:
        """前向传播"""
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=0)
            input_is_batch = False
        else:
            input_is_batch = True
        
        cache = []
        
        # Conv1 -> ReLU -> AvgPool (32 -> 28 -> 14)
        conv1 = convolution2d(X, self.W1, stride=1, padding=0) + self.b1
        cache.append(conv1.copy())
        relu1 = relu(conv1)
        pool1 = avg_pooling2d(relu1, pool_size=2, stride=2)
        
        # Conv2 -> ReLU -> AvgPool (14 -> 10 -> 5)
        conv2 = convolution2d(pool1, self.W2, stride=1, padding=0) + self.b2
        cache.append(conv2.copy())
        relu2 = relu(conv2)
        pool2 = avg_pooling2d(relu2, pool_size=2, stride=2)
        
        # Flatten
        flatten = pool2.reshape(pool2.shape[0], -1)
        cache.append(flatten.copy())
        
        # FC1 -> ReLU
        fc1 = flatten @ self.W3 + self.b3
        cache.append(fc1.copy())
        relu3 = relu(fc1)
        
        # FC2 -> ReLU
        fc2 = relu3 @ self.W4 + self.b4
        cache.append(fc2.copy())
        relu4 = relu(fc2)
        
        # FC3 (输出层)
        output = relu4 @ self.W5 + self.b5
        cache.append(output.copy())
        
        # Softmax
        output = softmax(output, axis=-1)
        
        if not input_is_batch:
            output = output[0]
        
        return output, cache
    
    def backward(self, y_true: np.ndarray, output: np.ndarray, 
                 cache: List, learning_rate: float = 0.01) -> dict:
        """反向传播"""
        m = y_true.shape[0]
        
        # 获取缓存
        fc3_out, fc2_out, fc1_out, flatten = cache[5], cache[4], cache[3], cache[2]
        
        # 输出层梯度 (Cross-Entropy with Softmax)
        dZ5 = output - y_true
        
        # FC3 梯度
        dW5 = (flatten.T @ dZ5) / m
        db5 = np.sum(dZ5, axis=0) / m
        
        # FC2 梯度
        dA4 = dZ5 @ self.W5.T
        dZ4 = dA4 * (fc2_out > 0)
        dW4 = (fc1_out.T @ dZ4) / m
        db4 = np.sum(dZ4, axis=0) / m
        
        # FC1 梯度
        dA3 = dZ4 @ self.W4.T
        dZ3 = dA3 * (fc1_out > 0)
        dW3 = (flatten.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0) / m
        
        # 更新权重
        self.W5 -= learning_rate * dW5
        self.b5 -= learning_rate * db5
        self.W4 -= learning_rate * dW4
        self.b4 -= learning_rate * db4
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        
        return {}
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32, 
            learning_rate: float = 0.1, verbose: bool = True):
        """训练模型"""
        n_samples = X_train.shape[0]
        
        # One-hot 编码
        if len(y_train.shape) == 1:
            y_train_onehot = np.eye(self.num_classes)[y_train]
        else:
            y_train_onehot = y_train
        
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                output, cache = self.forward(X_batch)
                
                # 计算损失
                loss = -np.mean(np.sum(y_batch * np.log(output + 1e-8), axis=1))
                acc = np.mean(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))
                
                # 反向传播
                self.backward(y_batch, output, cache, learning_rate)
                
                epoch_loss += loss
                epoch_acc += acc
                n_batches += 1
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        output, _ = self.forward(X, training=False)
        return np.argmax(output, axis=-1 if len(output.shape) > 1 else 0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        output, _ = self.forward(X, training=False)
        return output


class HOGExtractor:
    """HOG (Histogram of Oriented Gradients) 特征提取器"""
    
    def __init__(self, orientations: int = 9, pixels_per_cell: int = 8, 
                 cells_per_block: int = 2):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def compute_gradient(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算图像梯度"""
        gx = np.zeros_like(image, dtype=float)
        gy = np.zeros_like(image, dtype=float)
        
        gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gy[1:-1, :] = image[2:, :] - image[:-2, :]
        
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi
        orientation[orientation < 0] += 180
        
        return magnitude, orientation
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取 HOG 特征"""
        if len(image.shape) == 3:
            image = np.mean(image, axis=-1)
        
        magnitude, orientation = self.compute_gradient(image)
        
        h, w = image.shape
        n_cells_y = h // self.pixels_per_cell
        n_cells_x = w // self.pixels_per_cell
        
        # 计算每个细胞的直方图
        cell_histograms = np.zeros((n_cells_y, n_cells_x, self.orientations))
        
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                y_start = i * self.pixels_per_cell
                y_end = y_start + self.pixels_per_cell
                x_start = j * self.pixels_per_cell
                x_end = x_start + self.pixels_per_cell
                
                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ori = orientation[y_start:y_end, x_start:x_end]
                
                for m, o in zip(cell_mag.flatten(), cell_ori.flatten()):
                    bin_idx = int(o / 180 * self.orientations) % self.orientations
                    cell_histograms[i, j, bin_idx] += m
        
        # 块归一化
        features = []
        for i in range(n_cells_y - self.cells_per_block + 1):
            for j in range(n_cells_x - self.cells_per_block + 1):
                block = cell_histograms[i:i+self.cells_per_block, 
                                       j:j+self.cells_per_block].flatten()
                norm = np.sqrt(np.sum(block**2) + 1e-6)
                features.extend(block / norm)
        
        return np.array(features)


# =============================================================================
#                           示例
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("计算机视觉模型示例 - NumPy 实现")
    print("="*60)
    
    np.random.seed(42)
    
    # ========================================
    # 示例 1: LeNet-5
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: LeNet-5 NumPy 实现")
    print("-"*40)
    
    # 生成模拟数据
    n_samples = 100
    input_size = 32
    
    X = np.random.randn(n_samples, input_size, input_size, 1)
    y = np.random.randint(0, 10, n_samples)
    
    # 归一化
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    # 划分
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 训练
    model = LeNet5NumPy(input_size=input_size, num_classes=10)
    print("\n开始训练...")
    history = model.fit(X_train, y_train, epochs=3, batch_size=16, 
                       learning_rate=0.1, verbose=True)
    
    # 测试
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    # ========================================
    # 示例 2: HOG 特征提取
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: HOG 特征提取")
    print("-"*40)
    
    image = np.random.rand(64, 64)
    hog = HOGExtractor(orientations=9, pixels_per_cell=8, cells_per_block=2)
    features = hog.extract(image)
    
    print(f"输入图像尺寸: {image.shape}")
    print(f"HOG 特征维度: {len(features)}")
    
    # ========================================
    # 示例 3: 卷积操作测试
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: 卷积操作测试")
    print("-"*40)
    
    image = np.random.rand(32, 32, 3)
    kernel = np.random.randn(3, 3, 3, 8) * 0.1
    
    output = convolution2d(image, kernel, stride=1, padding=0)
    print(f"卷积输入: {image.shape}")
    print(f"卷积输出: {output.shape}")
    
    pool_out = max_pooling2d(output, pool_size=2, stride=2)
    print(f"池化输出: {pool_out.shape}")
    
    print("\n" + "="*60)
    print("计算机视觉模型示例完成!")
    print("="*60)
