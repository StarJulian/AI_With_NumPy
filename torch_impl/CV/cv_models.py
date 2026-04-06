"""
================================================================================
                        计算机视觉模型 - PyTorch 实现
================================================================================
包含:
    - LeNet-5: 经典卷积神经网络
    - AlexNet: ImageNet 2012 冠军
    - VGGNet: VGG-16/VGG-19
    - ResNet: 残差网络
    - MobileNet: 轻量级网络
    - YOLOv3: 目标检测
    - U-Net: 语义分割
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


# =============================================================================
#                           LeNet-5 实现
# =============================================================================
class LeNet5(nn.Module):
    """
    LeNet-5: 经典卷积神经网络 (1998)
    
    结构 (用于 32x32 输入):
        Conv1(1->6, 5x5) -> AvgPool -> Conv2(6->16, 5x5) -> AvgPool 
        -> FC(400->120) -> FC(120->84) -> FC(84->10)
        
    尺寸变化:
        32x32 -> Conv1(pad=2) -> 32x32 -> Pool -> 16x16
              -> Conv2(no pad) -> 12x12 -> Pool -> 6x6
        -> Flatten: 16*6*6 = 576
    """
    
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(LeNet5, self).__init__()
        
        # Conv1: 32x32 -> 28x28 (no padding) -> 14x14 (pool)
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=0)
        # Conv2: 14x14 -> 10x10 -> 5x5 (pool)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # FC1: 16 * 5 * 5 = 400 (原始 LeNet)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1 -> ReLU -> AvgPool
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        
        # Conv2 -> ReLU -> AvgPool
        x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def summary(self):
        """打印网络结构"""
        print("\n" + "="*60)
        print("LeNet-5 网络结构")
        print("="*60)
        print(f"{'层名称':<20} {'输出形状':<25} {'参数数量':<15}")
        print("-"*60)
        
        total_params = 0
        x = torch.randn(1, 1, 32, 32)
        
        layers = [
            ('conv1', self.conv1, [1, 1, 32, 32]),
            ('pool1', None, [1, 6, 14, 14]),
            ('conv2', self.conv2, [1, 6, 14, 14]),
            ('pool2', None, [1, 16, 5, 5]),
            ('fc1', self.fc1, [1, 400]),
            ('fc2', self.fc2, [1, 120]),
            ('fc3', self.fc3, [1, 84]),
        ]
        
        for name, layer, shape in layers:
            if layer is not None:
                params = sum(p.numel() for p in layer.parameters())
                total_params += params
            else:
                params = 0
            print(f"{name:<20} {str(shape):<25} {params:<15}")
        
        print("-"*60)
        print(f"{'总参数数量':<20} {'':<25} {total_params:,}")
        print("="*60)


# =============================================================================
#                           AlexNet 实现
# =============================================================================
class AlexNet(nn.Module):
    """
    AlexNet: ImageNet 2012 冠军 (2012)
    
    结构: 8层网络，包含5个卷积层和3个全连接层
    首次使用 ReLU、Dropout、GPU训练
    """
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
#                           VGGNet 实现
# =============================================================================
class VGGNet(nn.Module):
    """
    VGGNet: 牛津大学 Visual Geometry Group (2014)
    
    VGG-16: 13个卷积层 + 3个全连接层 = 16层
    VGG-19: 16个卷积层 + 3个全连接层 = 19层
    
    特点: 使用小卷积核(3x3)堆叠，深层网络
    """
    
    def __init__(self, config: str = 'VGG16', num_classes: int = 1000, dropout: float = 0.5):
        super(VGGNet, self).__init__()
        
        self.config_name = config
        
        # VGG 配置
        configs = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        self.features = self._make_layers(configs[config])
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        self._init_weights()
    
    def _make_layers(self, config: List):
        layers = []
        in_channels = 3
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
#                           ResNet 实现
# =============================================================================
class ResidualBlock(nn.Module):
    """ResNet 残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet: Deep Residual Learning (2015)
    
    核心思想: 恒等映射 (Identity Mapping)
    解决了深层网络梯度消失问题
    
    ResNet-18/34: BasicBlock
    ResNet-50/101/152: Bottleneck
    """
    
    def __init__(self, config: str = 'ResNet50', num_classes: int = 1000):
        super(ResNet, self).__init__()
        
        self.config_name = config
        
        # 不同深度的 ResNet 配置
        configs = {
            'ResNet18': ([2, 2, 2, 2], 64, 512, 'basic'),
            'ResNet34': ([3, 4, 6, 3], 64, 512, 'basic'),
            'ResNet50': ([3, 4, 6, 3], 64, 2048, 'bottleneck'),
            'ResNet101': ([3, 4, 23, 3], 64, 2048, 'bottleneck'),
            'ResNet152': ([3, 8, 36, 3], 64, 2048, 'bottleneck'),
        }
        
        layers, in_channels, out_channels, block_type = configs[config]
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(in_channels, 64, layers[0], stride=1, block_type=block_type)
        self.layer2 = self._make_layer(in_channels, 128, layers[1], stride=2, block_type=block_type)
        self.layer3 = self._make_layer(in_channels, 256, layers[2], stride=2, block_type=block_type)
        self.layer4 = self._make_layer(in_channels, 512, layers[3], stride=2, block_type=block_type)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int,
                    stride: int, block_type: str):
        
        if block_type == 'basic':
            block = ResidualBlock
            downsample = None
        else:  # bottleneck
            block = self._bottleneck_block
            downsample = None
        
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )
        
        layers = [block(in_channels, out_channels, stride, downsample)]
        
        for _ in range(1, blocks):
            if block_type == 'basic':
                layers.append(block(out_channels, out_channels))
            else:
                layers.append(self._bottleneck_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _bottleneck_block(self, in_channels: int, out_channels: int):
        """Bottleneck 结构: 1x1 -> 3x3 -> 1x1"""
        expansion = 4
        
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * expansion),
        )
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * expansion),
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# =============================================================================
#                           MobileNet V2 实现
# =============================================================================
class InvertedResidual(nn.Module):
    """MobileNetV2 倒置残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 6):
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # 扩展层 (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # 深度可分离卷积 (3x3 dw)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # 投影层 (1x1 conv)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2: Efficient Convolutional Neural Networks (2018)
    
    特点:
    - 倒置残差结构 (Inverted Residual)
    - 深度可分离卷积 (Depthwise Separable Convolution)
    - 线性瓶颈 (Linear Bottleneck)
    
    适合移动端和嵌入式设备
    """
    
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0):
        super(MobileNetV2, self).__init__()
        
        input_channel = 32
        last_channel = 1280
        
        # 配置: (expand_ratio, output_channel, num_blocks, stride)
        inverted_residual_setting = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        
        # 初始卷积层
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        
        features = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]
        
        # 倒置残差块
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        # 最终卷积层
        features.extend([
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True),
        ])
        
        self.features = nn.Sequential(*features)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
#                           YOLOv3 实现 (简化版)
# =============================================================================
class YOLOv3(nn.Module):
    """
    YOLOv3: You Only Look Once (2018)
    
    目标检测模型，单阶段检测器
    
    简化版本，包含:
    - Darknet-53 主干网络
    - 多尺度特征融合 (FPN)
    - YOLO Head (检测头)
    
    特点:
    - 实时检测
    - 多尺度预测
    - 锚框机制
    """
    
    def __init__(self, num_classes: int = 80, anchors: List = None):
        super(YOLOv3, self).__init__()
        
        self.num_classes = num_classes
        
        # 默认锚框 (COCO 数据集)
        if anchors is None:
            self.anchors = [
                # 小尺度 (13x13)
                [(10, 13), (16, 30), (33, 23)],
                # 中尺度 (26x26)
                [(30, 61), (62, 45), (59, 119)],
                # 大尺度 (52x52)
                [(116, 90), (156, 198), (373, 326)],
            ]
        else:
            self.anchors = anchors
        
        # Darknet-53 主干网络 (简化)
        self.backbone = self._build_backbone()
        
        # FPN 特征融合
        self.neck = self._build_neck()
        
        # 检测头 (都使用 1024 通道，因为 backbone 输出 1024)
        self.detect1 = self._build_detection_head(1024)  # 大尺度 13x13
        self.detect2 = self._build_detection_head(1024)  # 中尺度 26x26
        self.detect3 = self._build_detection_head(1024)  # 小尺度 52x52
    
    def _build_backbone(self):
        """Darknet-53 主干网络"""
        layers = []
        
        # Block 1
        layers.extend(self._darknet_block_layers(3, 32, 1))
        layers.extend(self._darknet_block_layers(32, 64, 2))
        
        # Block 2
        layers.extend(self._darknet_block_layers(64, 128, 8))
        
        # Block 3: 52x52 feature
        layers.extend(self._darknet_block_layers(128, 256, 8))
        self.route1_idx = len(layers)  # 记录跳跃连接位置
        
        # Block 4: 26x26 feature
        layers.extend(self._darknet_block_layers(256, 512, 8))
        self.route2_idx = len(layers)
        
        # Block 5: 13x13 feature
        layers.extend(self._darknet_block_layers(512, 1024, 4))
        
        return nn.Sequential(*layers)
    
    def _darknet_block(self, in_channels: int, out_channels: int, num_blocks: int):
        """Darknet 残差块 (返回 Sequential)"""
        layers = self._darknet_block_layers(in_channels, out_channels, num_blocks)
        return nn.Sequential(*layers)
    
    def _darknet_block_layers(self, in_channels: int, out_channels: int, num_blocks: int):
        """Darknet 残差块 (返回层列表)"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels//2, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(out_channels//2))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(nn.Conv2d(out_channels//2, out_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        return layers
    
    def _build_neck(self):
        """FPN 颈部网络"""
        return nn.Sequential(
            # 上采样和融合
            nn.Conv2d(1024, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
    
    def _build_detection_head(self, in_channels: int):
        """YOLO 检测头"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels//2, in_channels//4, 1, 1, 0),
            nn.BatchNorm2d(in_channels//4),
            nn.LeakyReLU(0.1, inplace=True),
            # 输出: (x, y, w, h, obj, cls1, cls2, ...)
            nn.Conv2d(in_channels//4, 3 * (5 + self.num_classes), 1, 1, 0),
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        返回:
            检测结果列表，每个尺度不同分辨率
        """
        # 主干网络
        features = self.backbone(x)  # (B, 1024, 13, 13)  for 416 input
        
        # 3个尺度的检测输出
        outputs = []
        
        # 大尺度 (13x13)
        out1 = self.detect1(features)
        outputs.append(out1)
        
        # 中尺度 (26x26) - 上采样
        up1 = F.interpolate(features, scale_factor=2, mode='nearest')
        out2 = self.detect2(up1)
        outputs.append(out2)
        
        # 小尺度 (52x52) - 继续上采样
        up2 = F.interpolate(up1, scale_factor=2, mode='nearest')
        out3 = self.detect3(up2)
        outputs.append(out3)
        
        return outputs


# =============================================================================
#                           U-Net 实现 (语义分割)
# =============================================================================
class DoubleConv(nn.Module):
    """U-Net 双卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """U-Net 下采样 (编码器)"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """U-Net 上采样 (解码器)"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: 上采样的特征
            x2: 来自编码器的跳跃连接
        """
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)
    
    结构: Encoder-Decoder + Skip Connections
    
    编码器 (4层下采样):
        64 -> 128 -> 256 -> 512
        
    解码器 (4层上采样):
        512 -> 256 -> 128 -> 64
        
    特点:
        - 编码器-解码器结构
        - 跳跃连接保留细节信息
        - 适用于医学图像分割
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: List[int] = None):
        super(UNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 编码器
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        
        # 解码器
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])
        
        # 输出层
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # 解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        logits = self.outc(x)
        
        return logits


# =============================================================================
#                           示例: 使用 CV 模型
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("计算机视觉模型示例 - PyTorch 实现")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # ========================================
    # 示例 1: LeNet-5
    # ========================================
    print("\n" + "-"*40)
    print("示例 1: LeNet-5")
    print("-"*40)
    
    lenet = LeNet5(num_classes=10)
    lenet.to(device)
    lenet.summary()
    
    # 测试前向传播
    x = torch.randn(1, 1, 32, 32).to(device)
    output = lenet(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 计算参数量
    params = sum(p.numel() for p in lenet.parameters())
    print(f"模型参数量: {params:,}")
    
    # ========================================
    # 示例 2: AlexNet
    # ========================================
    print("\n" + "-"*40)
    print("示例 2: AlexNet")
    print("-"*40)
    
    alexnet = AlexNet(num_classes=1000)
    alexnet.to(device)
    
    x = torch.randn(1, 3, 224, 224).to(device)
    output = alexnet(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    params = sum(p.numel() for p in alexnet.parameters())
    print(f"模型参数量: {params:,}")
    
    # ========================================
    # 示例 3: VGGNet
    # ========================================
    print("\n" + "-"*40)
    print("示例 3: VGGNet")
    print("-"*40)
    
    for config in ['VGG11', 'VGG16', 'VGG19']:
        vgg = VGGNet(config=config, num_classes=1000)
        vgg.to(device)
        
        x = torch.randn(1, 3, 224, 224).to(device)
        output = vgg(x)
        
        params = sum(p.numel() for p in vgg.parameters())
        print(f"{config}: 输入{x.shape} -> 输出{output.shape}, 参数量: {params:,}")
    
    # ========================================
    # 示例 4: ResNet
    # ========================================
    print("\n" + "-"*40)
    print("示例 4: ResNet")
    print("-"*40)
    
    for config in ['ResNet18', 'ResNet34', 'ResNet50']:
        resnet = ResNet(config=config, num_classes=1000)
        resnet.to(device)
        
        x = torch.randn(1, 3, 224, 224).to(device)
        output = resnet(x)
        
        params = sum(p.numel() for p in resnet.parameters())
        print(f"{config}: 输入{x.shape} -> 输出{output.shape}, 参数量: {params:,}")
    
    # ========================================
    # 示例 5: MobileNetV2
    # ========================================
    print("\n" + "-"*40)
    print("示例 5: MobileNetV2")
    print("-"*40)
    
    mobilenet = MobileNetV2(num_classes=1000)
    mobilenet.to(device)
    
    x = torch.randn(1, 3, 224, 224).to(device)
    output = mobilenet(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    params = sum(p.numel() for p in mobilenet.parameters())
    print(f"模型参数量: {params:,} (轻量级)")
    
    # ========================================
    # 示例 6: YOLOv3
    # ========================================
    print("\n" + "-"*40)
    print("示例 6: YOLOv3 (目标检测)")
    print("-"*40)
    
    yolo = YOLOv3(num_classes=80)
    yolo.to(device)
    
    x = torch.randn(1, 3, 416, 416).to(device)
    outputs = yolo(x)
    print(f"输入形状: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"检测头 {i+1} 输出形状: {out.shape}")
    
    params = sum(p.numel() for p in yolo.parameters())
    print(f"模型参数量: {params:,}")
    
    # ========================================
    # 示例 7: U-Net
    # ========================================
    print("\n" + "-"*40)
    print("示例 7: U-Net (语义分割)")
    print("-"*40)
    
    unet = UNet(in_channels=3, out_channels=1)
    unet.to(device)
    
    x = torch.randn(1, 3, 256, 256).to(device)
    output = unet(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    params = sum(p.numel() for p in unet.parameters())
    print(f"模型参数量: {params:,}")
    
    # ========================================
    # 示例 8: 训练循环示例 (以 U-Net 为例)
    # ========================================
    print("\n" + "-"*40)
    print("示例 8: U-Net 训练循环示例")
    print("-"*40)
    
    # 模拟训练
    unet = UNet(in_channels=3, out_channels=1)
    unet.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    
    # 模拟数据
    print("开始模拟训练...")
    unet.train()
    
    for epoch in range(3):
        # 模拟一个 batch
        inputs = torch.randn(4, 3, 256, 256).to(device)
        targets = torch.randn(4, 1, 256, 256).to(device)
        
        optimizer.zero_grad()
        outputs = unet(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/3], Loss: {loss.item():.4f}")
    
    print("\n训练完成!")
    
    # 测试推理
    unet.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).to(device)
        output = unet(x)
        pred = torch.sigmoid(output)
        print(f"推理输出形状: {pred.shape}")
        print(f"预测值范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    
    print("\n" + "="*60)
    print("计算机视觉模型示例完成!")
    print("="*60)
