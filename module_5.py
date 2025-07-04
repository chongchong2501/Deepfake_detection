#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 第5段：训练和验证函数
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

# =============================================================================
# 第5段：训练和验证函数
# =============================================================================

# 改进的数据增强
def get_enhanced_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((144, 144)),  # 稍大一些然后裁剪
            transforms.RandomCrop((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# 焦点损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 改进的训练函数
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        output, _ = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        predicted = (output > 0.5).float()
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# 改进的验证函数
def validate_epoch(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            output, _ = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 收集预测和目标用于计算指标
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    # 计算AUC
    try:
        auc_score = roc_auc_score(all_targets, all_predictions)
    except:
        auc_score = 0.0
    
    return epoch_loss, epoch_acc, auc_score

# 学习率调度器
class CosineAnnealingWarmRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = epoch
        if epoch >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        super(CosineAnnealingWarmRestarts, self).step(epoch)

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

# 优化的训练主函数
def train_optimized_model(model, train_loader, val_loader, num_epochs=50, 
                         learning_rate=0.001, device='cuda', save_path='./models'):
    """
    优化的模型训练函数
    """
    # 使用焦点损失
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }
    
    best_val_auc = 0.0
    
    print(f"开始训练，共 {num_epochs} 个epoch")
    print(f"设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # 验证
        val_loss, val_acc, val_auc = validate_epoch(
            model, val_loader, criterion, device, epoch, num_epochs
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # 打印epoch结果
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'history': history
            }, f'{save_path}/best_model.pth')
            print(f'  ✅ 保存最佳模型 (AUC: {val_auc:.4f})')
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f'早停触发，在第 {epoch+1} 个epoch停止训练')
            break
        
        print('-' * 60)
    
    return history

print("✅ 优化后的训练和验证函数定义完成")