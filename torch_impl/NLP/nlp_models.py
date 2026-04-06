"""
================================================================================
                    PyTorch CV 模块
================================================================================
计算机视觉算法实现
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class LeNet5(nn.Module):
    """LeNet-5 手写数字识别网络"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class AlexNet(nn.Module):
    """AlexNet 图像分类网络"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGGNet(nn.Module):
    """VGG 网络"""
    
    def __init__(self, config: List[int], num_classes: int = 1000):
        super().__init__()
        
        layers = []
        in_channels = 3
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                in_channels = v
        
        self.features = nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet 残差网络"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MobileNetV1(nn.Module):
    """MobileNet V1 (深度可分离卷积)"""
    
    def __init__(self, num_classes: int = 1000, alpha: float = 1.0):
        super().__init__()
        
        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32 * alpha), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * alpha)),
            nn.ReLU6(inplace=True),
        )
        
        # 深度可分离卷积块
        def conv_dw(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
            )
        
        self.features = nn.Sequential(
            conv_dw(int(32*alpha), int(64*alpha), 1),
            conv_dw(int(64*alpha), int(128*alpha), 2),
            conv_dw(int(128*alpha), int(128*alpha), 1),
            conv_dw(int(128*alpha), int(256*alpha), 2),
            conv_dw(int(256*alpha), int(256*alpha), 1),
            conv_dw(int(256*alpha), int(512*alpha), 2),
            *[conv_dw(int(512*alpha), int(512*alpha), 1) for _ in range(5)],
            conv_dw(int(512*alpha), int(1024*alpha), 2),
            conv_dw(int(1024*alpha), int(1024*alpha), 1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024*alpha), num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class YOLOV1(nn.Module):
    """YOLOv1 简化版目标检测"""
    
    def __init__(self, num_classes: int = 20, S: int = 7, B: int = 2):
        super().__init__()
        
        self.S = S
        self.B = B
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            *[self._make_layer(512, 256, 512) for _ in range(4)],
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (5 * B + num_classes)),
        )
    
    def _make_layer(self, in_ch: int, mid_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return x


class UNet(nn.Module):
    """U-Net 语义分割网络"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
            )
        
        # 编码器
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # 中间层
        self.middle = conv_block(512, 1024)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # 中间
        m = self.middle(self.pool(e4))
        
        # 解码器
        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.final(d1)


if __name__ == "__main__":
    print("="*60)
    print("PyTorch CV 模块示例")
    print("="*60)
    
    # 测试 ResNet
    print("\nResNet:")
    model = ResNet(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 测试 U-Net
    print("\nU-Net:")
    unet = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    y = unet(x)
    print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 测试 YOLO
    print("\nYOLOv1:")
    yolo = YOLOV1(num_classes=20, S=7)
    x = torch.randn(1, 3, 448, 448)
    y = yolo(x)
    print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
    
    print("\n" + "="*60)
