#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 第8段：模型评估
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

# =============================================================================
# 第8段：模型评估
# =============================================================================

import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 优化的评估函数
def evaluate_model_optimized(model, test_loader, criterion, device, save_attention=True):
    """
    优化的模型评估函数，包含更全面的指标和分析
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    all_scores = []
    all_attention_weights = []
    inference_times = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='模型评估中')
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            # 记录推理时间
            start_time = time.time()
            
            # 前向传播
            if hasattr(model, 'forward') and len(inspect.signature(model.forward).parameters) > 1:
                outputs, attention_weights = model(inputs)
                if save_attention:
                    all_attention_weights.append(attention_weights.cpu().numpy())
            else:
                outputs = model(inputs)
                attention_weights = None
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            outputs = outputs.squeeze()
            
            # 计算损失
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # 收集预测和目标
            preds = (outputs > 0.5).float().cpu().numpy()
            all_predictions.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_time': f'{np.mean(inference_times):.3f}s'
            })
    
    # 计算平均损失和推理时间
    test_loss = running_loss / len(test_loader.dataset)
    avg_inference_time = np.mean(inference_times)
    
    return {
        'loss': test_loss,
        'predictions': all_predictions,
        'targets': all_targets,
        'scores': all_scores,
        'attention_weights': all_attention_weights,
        'avg_inference_time': avg_inference_time,
        'total_inference_time': sum(inference_times)
    }

# 计算全面的评估指标
def calculate_comprehensive_metrics(predictions, targets, scores):
    """
    计算更全面的评估指标
    """
    # 基础指标
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    # ROC和PR曲线指标
    if len(set(targets)) > 1:
        auc_roc = roc_auc_score(targets, scores)
        precision_vals, recall_vals, _ = precision_recall_curve(targets, scores)
        auc_pr = average_precision_score(targets, scores)
    else:
        auc_roc = 0.0
        auc_pr = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # 额外指标
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 负预测值
    balanced_accuracy = (recall + specificity) / 2
    
    # 打印详细结果
    print("=" * 60)
    print("📊 模型评估结果")
    print("=" * 60)
    print(f"🎯 准确率 (Accuracy): {accuracy:.4f}")
    print(f"🎯 平衡准确率 (Balanced Accuracy): {balanced_accuracy:.4f}")
    print(f"🔍 精确率 (Precision): {precision:.4f}")
    print(f"🔍 召回率 (Recall/Sensitivity): {recall:.4f}")
    print(f"🔍 特异性 (Specificity): {specificity:.4f}")
    print(f"🔍 负预测值 (NPV): {npv:.4f}")
    print(f"⚖️ F1分数: {f1:.4f}")
    print(f"📈 AUC-ROC: {auc_roc:.4f}")
    print(f"📈 AUC-PR: {auc_pr:.4f}")
    print("=" * 60)
    
    # 混淆矩阵详情
    print("\n📋 混淆矩阵详情:")
    print(f"真阴性 (TN): {tn}")
    print(f"假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn}")
    print(f"真阳性 (TP): {tp}")
    
    # 分类报告
    print("\n📊 详细分类报告:")
    print(classification_report(targets, predictions, 
                             target_names=['真实视频', '伪造视频'],
                             digits=4))
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'npv': npv,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

# 增强的可视化函数
def plot_enhanced_confusion_matrix(cm, save_path=None):
    """
    绘制增强的混淆矩阵
    """
    plt.figure(figsize=(10, 8))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建标注
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
    
    annotations = np.array(annotations).reshape(cm.shape)
    
    # 绘制热力图
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                cbar_kws={'label': '样本数量'},
                xticklabels=['真实视频', '伪造视频'], 
                yticklabels=['真实视频', '伪造视频'])
    
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12, fontweight='bold')
    plt.title('混淆矩阵 (数量和百分比)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩阵已保存到: {save_path}")
    
    plt.show()

# 绘制ROC和PR曲线
def plot_roc_pr_curves(targets, scores, save_path=None):
    """
    同时绘制ROC曲线和PR曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('假阳性率 (FPR)', fontweight='bold')
    ax1.set_ylabel('真阳性率 (TPR)', fontweight='bold')
    ax1.set_title('ROC曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # PR曲线
    precision_vals, recall_vals, _ = precision_recall_curve(targets, scores)
    pr_auc = average_precision_score(targets, scores)
    
    ax2.plot(recall_vals, precision_vals, color='darkgreen', lw=2,
             label=f'PR曲线 (AUC = {pr_auc:.4f})')
    ax2.axhline(y=np.mean(targets), color='navy', linestyle='--', alpha=0.8,
                label=f'基线 ({np.mean(targets):.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('召回率 (Recall)', fontweight='bold')
    ax2.set_ylabel('精确率 (Precision)', fontweight='bold')
    ax2.set_title('精确率-召回率曲线', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC和PR曲线已保存到: {save_path}")
    
    plt.show()

# 注意力权重可视化
def visualize_attention_weights(attention_weights, save_path=None, num_samples=5):
    """
    可视化注意力权重
    """
    if not attention_weights:
        print("⚠️ 没有注意力权重数据")
        return
    
    # 选择前几个样本进行可视化
    num_samples = min(num_samples, len(attention_weights))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        weights = attention_weights[i].squeeze()
        if len(weights.shape) > 1:
            weights = weights.mean(axis=0)  # 如果是多头注意力，取平均
        
        axes[i].bar(range(len(weights)), weights, alpha=0.7)
        axes[i].set_title(f'样本 {i+1} 的注意力权重分布')
        axes[i].set_xlabel('帧序号')
        axes[i].set_ylabel('注意力权重')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        print(f"✅ 注意力权重可视化已保存到: {save_path}")
    
    plt.show()

print("✅ 优化的模型评估模块已定义")