
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import warnings
import gc
import json
import time
import sys
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

# PyTorch相关
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# 机器学习指标
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split

# 数据增强
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("警告: albumentations未安装，将使用基础数据增强")

# =============================================================================
# 全局配置和工具函数
# =============================================================================

def set_seed(seed=42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 创建必要的目录
for dir_name in ['./data', './models', './logs', './results', './results/evaluation']:
    os.makedirs(dir_name, exist_ok=True)

# 检查是否在Kaggle环境中
IS_KAGGLE = os.path.exists('/kaggle')
BASE_DATA_DIR = '/kaggle/input/ff-c23/FaceForensics++_C23' if IS_KAGGLE else './FaceForensics++_C23'

print(f"环境: {'Kaggle' if IS_KAGGLE else '本地'}")
print(f"数据基础路径: {BASE_DATA_DIR}")
print("✅ 环境设置完成")

# =============================================================================
# 数据处理模块
# =============================================================================

def extract_frames_memory_efficient(video_path, max_frames=24, target_size=(160, 160),
                                   quality_threshold=30, skip_frames=2):
    """内存友好的帧提取函数"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return frames

    # 均匀采样策略
    if total_frames <= max_frames:
        frame_indices = list(range(0, total_frames, skip_frames))
    else:
        step = max(1, total_frames // max_frames)
        frame_indices = list(range(0, total_frames, step))[:max_frames]

    frame_count = 0
    for frame_idx in frame_indices:
        if frame_count >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 简化质量检测
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            quality = cv2.Laplacian(gray, cv2.CV_64F).var()

            if quality > quality_threshold:
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
                frame_count += 1

    cap.release()

    # 如果帧数不足，重复最后一帧
    while len(frames) < max_frames and len(frames) > 0:
        frames.append(frames[-1].copy())

    return frames[:max_frames]

def process_videos_simple(base_data_dir, max_videos_per_class=80, max_frames=24):
    """简化的视频处理函数"""
    data_list = []
    fake_methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']

    print("开始处理真实视频...")
    # 处理真实视频
    original_dir = os.path.join(base_data_dir, 'original')
    if os.path.exists(original_dir):
        video_files = [f for f in os.listdir(original_dir)
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if len(video_files) > max_videos_per_class:
            video_files = random.sample(video_files, max_videos_per_class)

        print(f"找到 {len(video_files)} 个真实视频")

        for video_file in tqdm(video_files, desc="处理真实视频"):
            try:
                video_path = os.path.join(original_dir, video_file)
                frames = extract_frames_memory_efficient(video_path, max_frames)
                
                if len(frames) >= max_frames // 2:  # 至少要有一半的帧
                    data_list.append({
                        'video_path': video_path,
                        'frames': frames,
                        'label': 0,  # 真实视频
                        'method': 'original'
                    })
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {e}")
                continue

    # 处理伪造视频
    print("开始处理伪造视频...")
    for method in fake_methods:
        method_dir = os.path.join(base_data_dir, method)
        if os.path.exists(method_dir):
            video_files = [f for f in os.listdir(method_dir)
                          if f.endswith(('.mp4', '.avi', '.mov'))]
            
            if len(video_files) > max_videos_per_class:
                video_files = random.sample(video_files, max_videos_per_class)

            print(f"处理 {method}: {len(video_files)} 个视频")

            for video_file in tqdm(video_files, desc=f"处理{method}"):
                try:
                    video_path = os.path.join(method_dir, video_file)
                    frames = extract_frames_memory_efficient(video_path, max_frames)
                    
                    if len(frames) >= max_frames // 2:
                        data_list.append({
                            'video_path': video_path,
                            'frames': frames,
                            'label': 1,  # 伪造视频
                            'method': method
                        })
                except Exception as e:
                    print(f"处理视频 {video_file} 时出错: {e}")
                    continue

    print(f"\n✅ 数据处理完成，共处理 {len(data_list)} 个视频")
    return data_list

def create_dataset_split(data_list, test_size=0.2, val_size=0.1):
    """创建数据集划分"""
    # 分离真实和伪造数据
    real_data = [item for item in data_list if item['label'] == 0]
    fake_data = [item for item in data_list if item['label'] == 1]
    
    print(f"真实视频: {len(real_data)} 个")
    print(f"伪造视频: {len(fake_data)} 个")
    
    # 分别划分真实和伪造数据
    real_train, real_temp = train_test_split(real_data, test_size=test_size+val_size, random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=test_size/(test_size+val_size), random_state=42)
    
    fake_train, fake_temp = train_test_split(fake_data, test_size=test_size+val_size, random_state=42)
    fake_val, fake_test = train_test_split(fake_temp, test_size=test_size/(test_size+val_size), random_state=42)
    
    # 合并数据
    train_data = real_train + fake_train
    val_data = real_val + fake_val
    test_data = real_test + fake_test
    
    # 打乱数据
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def save_dataset_to_csv(data_list, filename):
    """将数据集保存为CSV文件"""
    df_data = []
    for item in data_list:
        df_data.append({
            'video_path': item['video_path'],
            'label': item['label'],
            'method': item['method'],
            'num_frames': len(item['frames'])
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"数据集已保存到: {filename}")
    return df

# =============================================================================
# 数据集类定义
# =============================================================================

class DeepfakeVideoDataset(Dataset):
    """深度伪造视频数据集类"""
    
    def __init__(self, csv_file=None, data_list=None, transform=None, max_frames=24):
        if csv_file is not None:
            self.df = pd.read_csv(csv_file)
            self.data_list = None
        elif data_list is not None:
            self.data_list = data_list
            self.df = None
        else:
            raise ValueError("必须提供csv_file或data_list")
            
        self.transform = transform
        self.max_frames = max_frames
    
    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.data_list is not None:
            # 直接从内存中的数据列表获取
            item = self.data_list[idx]
            frames = item['frames']
            label = item['label']
        else:
            # 从CSV文件获取路径并重新提取帧
            row = self.df.iloc[idx]
            video_path = row['video_path']
            label = row['label']
            frames = extract_frames_memory_efficient(video_path, self.max_frames)
        
        # 确保有足够的帧
        if len(frames) == 0:
            # 创建黑色帧作为fallback
            frames = [np.zeros((160, 160, 3), dtype=np.uint8) for _ in range(self.max_frames)]
        
        while len(frames) < self.max_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((160, 160, 3), dtype=np.uint8))
        
        frames = frames[:self.max_frames]
        
        # 应用变换
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        else:
            # 默认变换
            frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
        
        # 堆叠帧 (T, C, H, W)
        video_tensor = torch.stack(frames)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return video_tensor, label_tensor

# =============================================================================
# 模型定义
# =============================================================================

class OptimizedDeepfakeDetector(nn.Module):
    """优化的深度伪造检测模型"""
    
    def __init__(self, backbone='resnet50', hidden_dim=512, num_layers=2, 
                 dropout=0.3, use_attention=True):
        super(OptimizedDeepfakeDetector, self).__init__()
        
        self.use_attention = use_attention
        
        # 特征提取器
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 时序建模
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # 双向LSTM
        
        # 注意力机制
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames = x.shape[:2]
        
        # 重塑为 (batch_size * num_frames, channels, height, width)
        x = x.view(-1, *x.shape[2:])
        
        # 特征提取
        features = self.backbone(x)  # (batch_size * num_frames, feature_dim)
        
        # 重塑回时序格式
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(features)  # (batch_size, num_frames, hidden_dim*2)
        
        # 注意力机制
        attention_weights = None
        if self.use_attention:
            attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # 全局平均池化
            pooled = attended_out.mean(dim=1)  # (batch_size, hidden_dim*2)
        else:
            # 简单的全局平均池化
            pooled = lstm_out.mean(dim=1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output.squeeze(-1), attention_weights

# =============================================================================
# 损失函数和工具类
# =============================================================================

class FocalLoss(nn.Module):
    """焦点损失函数 - 解决类别不平衡问题"""
    
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

class EarlyStopping:
    """早停机制"""
    
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

# =============================================================================
# 数据变换
# =============================================================================

def get_transforms(mode='train', image_size=160):
    """获取数据变换"""
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

# =============================================================================
# 训练和验证函数
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc='Training', leave=False)

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                output, _ = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        predicted = (output > 0.5).float()
        total += target.size(0)
        correct += (predicted == target).sum().item()

        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except:
        auc_score = 0.0

    return avg_loss, accuracy, auc_score

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
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
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except:
        auc_score = 0.0

    return avg_loss, accuracy, auc_score

# =============================================================================
# 评估函数
# =============================================================================

def evaluate_model_optimized(model, test_loader, criterion, device):
    """优化的模型评估函数"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_scores = []
    
    inference_times = []
    
    print("🚀 开始模型评估...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="评估进度")):
            data, target = data.to(device), target.to(device)
            
            # 记录推理时间
            start_time = time.time()
            output, attention_weights = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 收集预测结果
            predictions = (output > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(output.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    
    print(f"✅ 评估完成")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"平均推理时间: {avg_inference_time*1000:.2f} ms/batch")
    
    return {
        'loss': avg_loss,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'scores': np.array(all_scores),
        'avg_inference_time': avg_inference_time,
        'total_inference_time': total_inference_time
    }

def calculate_comprehensive_metrics(predictions, targets, scores):
    """计算全面的评估指标"""
    # 基础指标
    accuracy = accuracy_score(targets, predictions)
    balanced_acc = balanced_accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # 特异性和负预测值
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # AUC指标
    try:
        auc_roc = roc_auc_score(targets, scores)
    except:
        auc_roc = 0.0
    
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(targets, scores)
        auc_pr = auc(recall_curve, precision_curve)
    except:
        auc_pr = 0.0
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'npv': npv,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def plot_enhanced_confusion_matrix(cm, save_path):
    """绘制增强的混淆矩阵"""
    plt.figure(figsize=(10, 8))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建标签
    labels = np.array([[
        f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
        for j in range(cm.shape[1])
    ] for i in range(cm.shape[0])])
    
    # 绘制热图
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=['真实', '伪造'],
                yticklabels=['真实', '伪造'],
                cbar_kws={'label': '样本数量'})
    
    plt.title('增强混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    
    # 添加统计信息
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    stats_text = f'准确率: {accuracy:.3f}\n精确率: {precision:.3f}\n召回率: {recall:.3f}\nF1分数: {f1:.3f}'
    plt.text(2.1, 0.5, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")

def plot_roc_pr_curves(targets, scores, save_path):
    """绘制ROC和PR曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('假正率')
    ax1.set_ylabel('真正率')
    ax1.set_title('ROC曲线')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # PR曲线
    precision_curve, recall_curve, _ = precision_recall_curve(targets, scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    ax2.plot(recall_curve, precision_curve, color='darkgreen', lw=2,
             label=f'PR曲线 (AUC = {pr_auc:.4f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('召回率')
    ax2.set_ylabel('精确率')
    ax2.set_title('精确率-召回率曲线')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC/PR曲线已保存到: {save_path}")

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数 - 演示完整的训练和评估流程"""
    print("🚀 深度伪造检测模型启动...")
    print("=" * 60)
    
    # 如果需要处理数据（首次运行）
    if not os.path.exists('./data/train.csv'):
        print("📁 开始数据处理...")
        data_list = process_videos_simple(BASE_DATA_DIR, max_videos_per_class=50, max_frames=24)
        
        if len(data_list) == 0:
            print("❌ 未找到数据，请检查数据路径")
            return
        
        train_data, val_data, test_data = create_dataset_split(data_list)
        
        # 保存数据集
        save_dataset_to_csv(train_data, './data/train.csv')
        save_dataset_to_csv(val_data, './data/val.csv')
        save_dataset_to_csv(test_data, './data/test.csv')
        
        print(f"训练集: {len(train_data)} 个样本")
        print(f"验证集: {len(val_data)} 个样本")
        print(f"测试集: {len(test_data)} 个样本")
    
    # 创建数据加载器
    print("📊 创建数据加载器...")
    
    # 使用CSV文件创建数据集（节省内存）
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = DeepfakeVideoDataset('./data/train.csv', transform=train_transform)
    val_dataset = DeepfakeVideoDataset('./data/val.csv', transform=val_transform)
    test_dataset = DeepfakeVideoDataset('./data/test.csv', transform=val_transform)
    
    batch_size = 8 if torch.cuda.is_available() else 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"批次大小: {batch_size}")
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    # 创建模型
    print("🤖 创建模型...")
    model = OptimizedDeepfakeDetector(
        backbone='resnet18',  # 使用更轻量的backbone
        hidden_dim=256,
        num_layers=1,
        dropout=0.3,
        use_attention=True
    ).to(device)
    
    # 损失函数和优化器
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=5)
    
    # 混合精度训练
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n🎯 开始训练...")
    num_epochs = 20
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # 验证
        val_loss, val_acc, val_auc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%, AUC: {train_auc:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%, AUC: {val_auc:.4f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_loss': val_loss
            }, './models/best_model.pth')
            print(f"💾 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f"🛑 早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    # 加载最佳模型进行测试
    print("\n📊 加载最佳模型进行测试...")
    checkpoint = torch.load('./models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估模型
    print("\n🔍 开始最终评估...")
    eval_results = evaluate_model_optimized(model, test_loader, criterion, device)
    metrics = calculate_comprehensive_metrics(
        eval_results['predictions'], 
        eval_results['targets'], 
        eval_results['scores']
    )
    
    # 打印结果
    print("\n📈 最终评估结果:")
    print(f"测试损失: {eval_results['loss']:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"平衡准确率: {metrics['balanced_accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    
    # 生成可视化
    print("\n📊 生成评估图表...")
    plot_enhanced_confusion_matrix(
        metrics['confusion_matrix'], 
        './results/evaluation/confusion_matrix.png'
    )
    plot_roc_pr_curves(
        eval_results['targets'], 
        eval_results['scores'], 
        './results/evaluation/roc_pr_curves.png'
    )
    
    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'backbone': 'resnet18',
            'hidden_dim': 256,
            'num_layers': 1,
            'dropout': 0.3,
            'use_attention': True
        },
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4
        },
        'metrics': {
            'test_loss': float(eval_results['loss']),
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'auc_roc': float(metrics['auc_roc']),
            'auc_pr': float(metrics['auc_pr']),
            'specificity': float(metrics['specificity']),
            'npv': float(metrics['npv'])
        },
        'performance': {
            'avg_inference_time_ms': eval_results['avg_inference_time'] * 1000,
            'total_inference_time': eval_results['total_inference_time']
        }
    }
    
    with open('./results/evaluation/final_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n🎉 训练和评估完成！")
    print("📁 结果已保存到 ./results/evaluation/ 目录")
    print("=" * 60)

if __name__ == "__main__":
    main()