#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 第9段：完整的模型评估和结果分析
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

# =============================================================================
# Kaggle 兼容的深度伪造检测模型评估脚本
# =============================================================================

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, auc, average_precision_score, precision_recall_curve
)
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义数据转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集类定义
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
                if self.transform:
                    dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    frames.append(self.transform(dummy_frame))
                else:
                    frames.append(torch.zeros(3, 224, 224))
        
        # 将帧堆叠成张量
        frames_tensor = torch.stack(frames[:self.max_frames])
        
        return frames_tensor, torch.tensor(label, dtype=torch.float32)

# 模型定义
class OptimizedDeepfakeDetector(nn.Module):
    """优化的深度伪造检测模型"""
    def __init__(self, backbone='resnet50', hidden_dim=512, num_layers=2, dropout=0.3, use_attention=True):
        super().__init__()
        
        # 使用预训练的ResNet作为特征提取器
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
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
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, 8, dropout=dropout, batch_first=True)
        
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
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            pooled = torch.mean(attn_out, dim=1)
        else:
            pooled = torch.mean(lstm_out, dim=1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output.squeeze(-1)

# 优化的模型评估函数
def evaluate_model_optimized(model, test_loader, criterion, device):
    """优化的模型评估函数"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    scores = []
    
    total_inference_time = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 记录推理时间
            import time
            start_time = time.time()
            
            # 前向传播
            outputs = model(inputs)
            
            batch_time = time.time() - start_time
            total_inference_time += batch_time
            
            if len(outputs.shape) > 1:
                outputs = outputs.squeeze()
            
            # 计算损失
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # 收集预测结果
            batch_preds = (outputs > 0.5).float().cpu().numpy()
            batch_targets = labels.cpu().numpy()
            batch_scores = outputs.cpu().numpy()
            
            predictions.extend(batch_preds)
            targets.extend(batch_targets)
            scores.extend(batch_scores)
    
    avg_loss = running_loss / len(test_loader.dataset)
    avg_inference_time = total_inference_time / len(test_loader.dataset)
    
    return {
        'loss': avg_loss,
        'predictions': np.array(predictions),
        'targets': np.array(targets),
        'scores': np.array(scores),
        'total_inference_time': total_inference_time,
        'avg_inference_time': avg_inference_time
    }

# 计算全面的评估指标
def calculate_comprehensive_metrics(predictions, targets, scores):
    """计算全面的评估指标"""
    # 基础指标
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # 额外指标
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 负预测值
    balanced_accuracy = (recall + specificity) / 2
    
    # AUC指标
    try:
        auc_roc = roc_auc_score(targets, scores)
        auc_pr = average_precision_score(targets, scores)
    except:
        auc_roc = 0.0
        auc_pr = 0.0
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'npv': npv,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

# 增强的混淆矩阵可视化
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
    
    # 绘制热力图
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=['真实', '伪造'], 
                yticklabels=['真实', '伪造'],
                cbar_kws={'label': '样本数量'})
    
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('增强混淆矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ROC和PR曲线
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
    ax1.set_xlabel('假阳性率')
    ax1.set_ylabel('真阳性率')
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

# 主评估流程
print("🔄 创建优化的测试数据加载器...")

# 使用更大的批次大小以提高效率
optimized_batch_size = 16 if torch.cuda.is_available() else 8

test_dataset = DeepfakeVideoDataset(
    csv_file='./data/val.csv',
    transform=transform,
    max_frames=30
)

test_loader = DataLoader(
    test_dataset,
    batch_size=optimized_batch_size,
    shuffle=False,
    num_workers=2,  # 减少worker数量以适应Kaggle环境
    pin_memory=True
)

print(f"📊 测试集大小: {len(test_dataset)} 个样本")
print(f"🔧 批次大小: {optimized_batch_size}")
print(f"🔧 批次数量: {len(test_loader)}")

# 加载最佳模型
print("\n🤖 加载训练好的模型...")

try:
    checkpoint = torch.load('./models/best_model.pth', 
                          map_location=device, 
                          weights_only=False)
except TypeError:
    checkpoint = torch.load('./models/best_model.pth', map_location=device)

# 创建模型并加载权重
model = OptimizedDeepfakeDetector(
    backbone='resnet50',
    hidden_dim=512,
    num_layers=2,
    dropout=0.3,
    use_attention=True
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ 模型加载成功")
print(f"📈 最佳验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")
print(f"🔄 训练轮数: {checkpoint.get('epoch', 'N/A')}")

# 记录评估开始时间
eval_start_time = datetime.now()
print(f"\n⏰ 评估开始时间: {eval_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# 执行优化的模型评估
print("\n🚀 开始执行模型评估...")
print("=" * 60)

criterion = nn.BCELoss()

# 执行评估
eval_results = evaluate_model_optimized(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device=device
)

# 计算全面的评估指标
metrics = calculate_comprehensive_metrics(
    predictions=eval_results['predictions'],
    targets=eval_results['targets'],
    scores=eval_results['scores']
)

# 性能分析
eval_end_time = datetime.now()
total_eval_time = (eval_end_time - eval_start_time).total_seconds()

print(f"\n⏱️ 评估性能分析:")
print(f"总评估时间: {total_eval_time:.2f} 秒")
print(f"平均每样本推理时间: {eval_results['avg_inference_time']*1000:.2f} ms")
print(f"推理吞吐量: {len(test_dataset)/eval_results['total_inference_time']:.1f} 样本/秒")

# 打印详细评估结果
print(f"\n📊 详细评估结果:")
print(f"测试损失: {eval_results['loss']:.4f}")
print(f"准确率: {metrics['accuracy']:.4f}")
print(f"平衡准确率: {metrics['balanced_accuracy']:.4f}")
print(f"精确率: {metrics['precision']:.4f}")
print(f"召回率: {metrics['recall']:.4f}")
print(f"特异性: {metrics['specificity']:.4f}")
print(f"F1分数: {metrics['f1']:.4f}")
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"AUC-PR: {metrics['auc_pr']:.4f}")
print(f"负预测值: {metrics['npv']:.4f}")

# 创建结果目录
os.makedirs('./results/evaluation', exist_ok=True)

# 绘制增强的可视化图表
print("\n📊 生成评估图表...")

# 1. 增强的混淆矩阵
plot_enhanced_confusion_matrix(
    cm=metrics['confusion_matrix'],
    save_path='./results/evaluation/enhanced_confusion_matrix.png'
)

# 2. ROC和PR曲线
plot_roc_pr_curves(
    targets=eval_results['targets'],
    scores=eval_results['scores'],
    save_path='./results/evaluation/roc_pr_curves.png'
)

# 保存详细的评估报告
detailed_report = {
    'evaluation_info': {
        'timestamp': eval_start_time.isoformat(),
        'model_path': './models/best_model.pth',
        'test_dataset_size': len(test_dataset),
        'batch_size': optimized_batch_size
    },
    'performance_metrics': {
        'test_loss': float(eval_results['loss']),
        'accuracy': float(metrics['accuracy']),
        'balanced_accuracy': float(metrics['balanced_accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'specificity': float(metrics['specificity']),
        'f1_score': float(metrics['f1']),
        'auc_roc': float(metrics['auc_roc']),
        'auc_pr': float(metrics['auc_pr']),
        'npv': float(metrics['npv'])
    },
    'confusion_matrix': {
        'true_negative': int(metrics['tn']),
        'false_positive': int(metrics['fp']),
        'false_negative': int(metrics['fn']),
        'true_positive': int(metrics['tp'])
    },
    'performance_analysis': {
        'total_evaluation_time_seconds': total_eval_time,
        'average_inference_time_ms': eval_results['avg_inference_time'] * 1000,
        'throughput_samples_per_second': len(test_dataset) / eval_results['total_inference_time']
    }
}

# 保存JSON格式的详细报告
with open('./results/evaluation/detailed_evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(detailed_report, f, indent=2, ensure_ascii=False)

# 保存CSV格式的简要报告
summary_df = pd.DataFrame([{
    '评估时间': eval_start_time.strftime('%Y-%m-%d %H:%M:%S'),
    '测试损失': f"{eval_results['loss']:.4f}",
    '准确率': f"{metrics['accuracy']:.4f}",
    '平衡准确率': f"{metrics['balanced_accuracy']:.4f}",
    '精确率': f"{metrics['precision']:.4f}",
    '召回率': f"{metrics['recall']:.4f}",
    'F1分数': f"{metrics['f1']:.4f}",
    'AUC-ROC': f"{metrics['auc_roc']:.4f}",
    'AUC-PR': f"{metrics['auc_pr']:.4f}",
    '推理时间(ms)': f"{eval_results['avg_inference_time']*1000:.2f}",
    '吞吐量(样本/秒)': f"{len(test_dataset)/eval_results['total_inference_time']:.1f}"
}])

summary_df.to_csv('./results/evaluation/evaluation_summary.csv', index=False, encoding='utf-8')

print("\n📁 评估结果已保存到:")
print("  📊 ./results/evaluation/enhanced_confusion_matrix.png")
print("  📈 ./results/evaluation/roc_pr_curves.png")
print("  📋 ./results/evaluation/detailed_evaluation_report.json")
print("  📊 ./results/evaluation/evaluation_summary.csv")

print("\n🎉 模型评估完成！")
print("=" * 60)