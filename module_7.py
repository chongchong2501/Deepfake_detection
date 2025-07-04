#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 第7段：执行训练循环
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

# =============================================================================
# 第7段：执行训练循环
# =============================================================================

# 添加必要的导入
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_auc_score, accuracy_score
from torchvision import transforms
import random
from collections import Counter

# 直接定义DeepfakeVideoDataset类（适用于Kaggle环境）
class DeepfakeVideoDataset(Dataset):
    """深度伪造检测数据集（Kaggle兼容版本）"""
    def __init__(self, csv_file, transform=None, max_frames=30, mode='train'):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.max_frames = max_frames
        self.mode = mode
        
        print(f"数据集初始化完成: {len(self.data)} 个样本 ({mode} 模式)")
        if 'label' in self.data.columns:
            print(f"真实视频: {len(self.data[self.data['label']==0])} 个")
            print(f"伪造视频: {len(self.data[self.data['label']==1])} 个")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 处理帧路径
        if 'frames' in row:
            try:
                frame_paths = eval(row['frames'])  # 将字符串转换为列表
                if isinstance(frame_paths, str):
                    frame_paths = [frame_paths]
            except:
                frame_paths = [row['frames']]
        elif 'frame_path' in row:
            frame_paths = [row['frame_path']]
        else:
            # 如果没有找到帧路径，创建随机帧
            frame_paths = []
        
        label = float(row['label']) if 'label' in row else 0.0
        
        # 加载帧
        frames = []
        for i, frame_path in enumerate(frame_paths[:self.max_frames]):
            try:
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (224, 224))
                    else:
                        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                else:
                    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            except:
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            frames.append(frame)
        
        # 如果帧数不足，用零填充或重复最后一帧
        while len(frames) < self.max_frames:
            if frames:
                frames.append(frames[-1].clone())
            else:
                # 如果没有帧，创建零张量
                if self.transform:
                    dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    frames.append(self.transform(dummy_frame))
                else:
                    frames.append(torch.zeros(3, 224, 224))
        
        # 将帧堆叠成张量
        frames_tensor = torch.stack(frames[:self.max_frames])
        
        return frames_tensor, torch.tensor(label, dtype=torch.float32)

# 定义简化的模型类（如果需要）
class OptimizedDeepfakeDetector(nn.Module):
    """优化的深度伪造检测模型"""
    def __init__(self, backbone='resnet50', hidden_dim=512, num_layers=2, dropout=0.3, num_heads=8):
        super().__init__()
        
        # 使用预训练的ResNet作为特征提取器
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()  # 移除最后的分类层
            feature_dim = 2048
        else:
            # 简化的CNN特征提取器
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            feature_dim = 256
        
        # 时序处理
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # 提取每帧特征
        x = x.view(-1, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM处理时序信息
        lstm_out, _ = self.lstm(features)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output.squeeze(-1)

# 定义损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# 早停机制
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(np.array(all_targets) > 0.5, np.array(all_preds) > 0.5) * 100
    auc = roc_auc_score(all_targets, all_preds)
    
    return avg_loss, accuracy, auc

# 验证函数
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(np.array(all_targets) > 0.5, np.array(all_preds) > 0.5) * 100
    auc = roc_auc_score(all_targets, all_preds)
    
    return avg_loss, accuracy, auc

# 定义训练配置（保持您的原始参数）
TRAIN_CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'patience': 10,
    'use_focal_loss': True,
    'use_mixed_precision': True
}

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据变换（保持您的原始配置）
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建必要的目录
os.makedirs('./models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# 创建优化的数据集和数据加载器
print("创建数据集和数据加载器...")

# 使用改进的数据增强
train_dataset = DeepfakeVideoDataset('./data/train.csv', transform=train_transform, max_frames=30)
val_dataset = DeepfakeVideoDataset('./data/val.csv', transform=val_transform, max_frames=30)

# ... existing code ...