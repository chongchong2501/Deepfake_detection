#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 第6段：模型训练
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

# =============================================================================
# 第6段：模型训练
# =============================================================================

import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# 焦点损失函数 - 解决类别不平衡问题
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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

# 改进的数据增强
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((144, 144)),  # 稍大一些，为裁剪留空间
    transforms.RandomCrop((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # 随机擦除
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 优化的训练函数
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # 创建进度条
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        if scaler is not None:
            with autocast():
                output, _ = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        predicted = (output > 0.5).float()
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 收集预测和目标用于AUC计算
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # 计算AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except:
        auc_score = 0.0
    
    return avg_loss, accuracy, auc_score

# 优化的验证函数
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output, _ = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # 计算准确率
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 收集预测和目标
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # 计算AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except:
        auc_score = 0.0
    
    return avg_loss, accuracy, auc_score

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 8,  # 增加批次大小
    'learning_rate': 1e-4,  # 降低学习率
    'num_epochs': 50,  # 增加训练轮数
    'weight_decay': 1e-4,
    'patience': 10,  # 早停耐心值
    'use_focal_loss': True,
    'use_mixed_precision': True,
    'gradient_clip': 1.0
}

print("✅ 优化的模型训练函数定义完成")
print(f"训练配置: {TRAIN_CONFIG}")