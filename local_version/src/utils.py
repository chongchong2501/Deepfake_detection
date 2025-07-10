# 工具函数和损失函数 - 本地RTX4070优化版本

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import time
import psutil
from config import config

class FocalLoss(nn.Module):
    """焦点损失函数 - 解决类别不平衡问题"""
    
    def __init__(self, alpha=None, gamma=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha or config.FOCAL_ALPHA
        self.gamma = gamma or config.FOCAL_GAMMA
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 使用 BCEWithLogitsLoss 以兼容混合精度训练
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        # 计算概率用于focal weight
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=None, min_delta=None, restore_best_weights=True):
        self.patience = patience or config.EARLY_STOPPING_PATIENCE
        self.min_delta = min_delta or config.EARLY_STOPPING_MIN_DELTA
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch or 0
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch or 0
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
                print(f"早停触发，恢复第{self.best_epoch}轮的最佳权重")
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
    
    def get_best_loss(self):
        return self.best_loss

class PerformanceMonitor:
    """性能监控器（增强版）"""
    
    def __init__(self):
        # 尝试集成新的内存管理器
        try:
            from memory_manager import get_memory_manager
            self.memory_manager = get_memory_manager()
            self.enhanced_monitoring = True
        except ImportError:
            self.memory_manager = None
            self.enhanced_monitoring = False
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.gpu_memory_peak = 0
        self.cpu_usage_history = []
        self.gpu_usage_history = []
    
    def update(self):
        if self.enhanced_monitoring and self.memory_manager:
            # 使用增强的内存管理器
            stats = self.memory_manager.get_memory_stats()
            self.cpu_usage_history.append(stats.cpu_percent)
            gpu_memory = stats.gpu_memory_allocated_gb
            self.gpu_memory_peak = max(self.gpu_memory_peak, gpu_memory)
        else:
            # 回退到原始实现
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage_history.append(cpu_percent)
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                self.gpu_memory_peak = max(self.gpu_memory_peak, gpu_memory)
    
    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        avg_cpu = np.mean(self.cpu_usage_history) if self.cpu_usage_history else 0
        
        stats = {
            'elapsed_time': elapsed_time,
            'avg_cpu_usage': avg_cpu,
            'peak_gpu_memory_gb': self.gpu_memory_peak
        }
        
        # 如果有增强监控，添加更多统计信息
        if self.enhanced_monitoring and self.memory_manager:
            current_stats = self.memory_manager.get_memory_stats()
            stats.update({
                'current_gpu_memory_gb': current_stats.gpu_memory_allocated_gb,
                'current_cpu_memory_gb': current_stats.cpu_memory_gb,
                'gpu_memory_percent': current_stats.gpu_memory_percent,
                'cpu_memory_percent': current_stats.cpu_memory_percent
            })
        elif torch.cuda.is_available():
            stats['current_gpu_memory_gb'] = torch.cuda.memory_allocated() / 1024**3
            stats['max_gpu_memory_gb'] = torch.cuda.max_memory_allocated() / 1024**3
        
        return stats
    
    def get_memory_suggestions(self):
        """获取内存优化建议"""
        if self.enhanced_monitoring and self.memory_manager:
            return self.memory_manager.get_optimization_suggestions()
        else:
            # 基础建议
            suggestions = []
            if self.gpu_memory_peak > 10:  # 假设12GB显卡
                suggestions.append("🟡 GPU内存使用较高，建议减小batch_size")
            if len(self.cpu_usage_history) > 0 and max(self.cpu_usage_history) > 90:
                suggestions.append("🟡 CPU使用率过高，建议减少num_workers")
            if not suggestions:
                suggestions.append("✅ 系统资源使用正常")
            return suggestions

def get_transforms(mode='train', image_size=None):
    """获取数据变换"""
    if image_size is None:
        image_size = config.FRAME_SIZE[0]  # 假设是正方形
    
    if mode == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def calculate_accuracy(outputs, targets, threshold=0.5):
    """计算准确率"""
    predictions = torch.sigmoid(outputs) > threshold
    targets_bool = targets > threshold
    correct = (predictions == targets_bool).float()
    return correct.mean().item() * 100

def calculate_auc(outputs, targets):
    """计算AUC分数"""
    try:
        from sklearn.metrics import roc_auc_score
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        return roc_auc_score(targets_np, probs)
    except Exception:
        return 0.0

def save_model_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, auc, filepath):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'accuracy': accuracy,
        'auc': auc,
        'config': {
            'backbone': config.BACKBONE,
            'hidden_dim': config.HIDDEN_DIM,
            'num_layers': config.NUM_LSTM_LAYERS,
            'dropout': config.DROPOUT,
            'use_attention': config.USE_ATTENTION,
            'max_frames': config.MAX_FRAMES,
            'frame_size': config.FRAME_SIZE
        }
    }
    
    torch.save(checkpoint, filepath)
    print(f"模型检查点已保存到: {filepath}")

def load_model_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """加载模型检查点"""
    checkpoint = torch.load(filepath, map_location=config.get_device())
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    accuracy = checkpoint.get('accuracy', 0.0)
    auc = checkpoint.get('auc', 0.0)
    
    print(f"模型检查点已加载: epoch={epoch}, loss={loss:.4f}, acc={accuracy:.2f}%, auc={auc:.4f}")
    
    return epoch, loss, accuracy, auc

def print_gpu_memory_info():
    """打印GPU内存信息（兼容旧版本）"""
    try:
        from memory_manager import print_memory_info
        print_memory_info()
    except ImportError:
        # 回退到原始实现
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"GPU内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB, 峰值: {max_allocated:.2f}GB")
        else:
            print("CUDA不可用")

def cleanup_gpu_memory():
    """清理GPU内存（兼容旧版本）"""
    try:
        from memory_manager import cleanup_memory
        return cleanup_memory()
    except ImportError:
        # 回退到原始实现
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return 0.0

def set_random_seed(seed=None):
    """设置随机种子"""
    if seed is None:
        seed = config.RANDOM_SEED
    
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"随机种子已设置为: {seed}")

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}分{seconds:.1f}秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}小时{int(minutes)}分{seconds:.1f}秒"

def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }