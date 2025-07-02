import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from data_preprocessing import DeepfakeVideoDataset, set_seed
from model import create_model

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='深度伪造检测模型评估')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--model_dir', type=str, default='./models', help='模型目录')
    parser.add_argument('--results_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lightweight'], help='模型类型')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--num_frames', type=int, default=30, help='每个视频的帧数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='评估')
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            # 前向传播
            if isinstance(model.forward(inputs), tuple):
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            
            outputs = outputs.squeeze()
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            
            # 收集预测和目标
            preds = (outputs > 0.5).float().cpu().numpy()
            all_predictions.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
    
    # 计算平均损失
    test_loss = running_loss / len(test_loader.dataset)
    
    return test_loss, all_predictions, all_targets, all_scores

# 计算并打印指标
def calculate_metrics(predictions, targets, scores):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    auc_score = roc_auc_score(targets, scores) if len(set(targets)) > 1 else 0.0
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(targets, predictions, target_names=['真实', '伪造']))
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': cm
    }

# 绘制混淆矩阵
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['真实', '伪造'], yticklabels=['真实', '伪造'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 绘制ROC曲线
def plot_roc_curve(targets, scores, save_path):
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 分析错误预测
def analyze_errors(predictions, targets, scores, dataset, save_dir):
    # 找出错误预测的样本
    errors = []
    for i in range(len(predictions)):
        if predictions[i] != targets[i]:
            errors.append({
                'index': i,
                'true_label': targets[i],
                'pred_label': predictions[i],
                'confidence': scores[i]
            })
    
    # 按置信度排序
    errors.sort(key=lambda x: abs(x['confidence'] - 0.5))
    
    # 保存错误分析结果
    error_df = pd.DataFrame(errors)
    error_df.to_csv(os.path.join(save_dir, 'error_analysis.csv'), index=False)
    
    print(f"错误预测样本数: {len(errors)}")
    print(f"错误分析已保存到 {os.path.join(save_dir, 'error_analysis.csv')}")
    
    return errors

# 主函数
def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据集
    test_dataset = DeepfakeVideoDataset(
        csv_file=os.path.join(args.data_dir, 'val.csv'),  # 使用验证集作为测试集
        transform=transform,
        max_frames=args.num_frames
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 加载模型
    model = create_model(model_type=args.model_type, device=device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型: {args.model_path}")
    
    # 定义损失函数
    criterion = nn.BCELoss()
    
    # 评估模型
    print("开始评估...")
    test_loss, predictions, targets, scores = evaluate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}")
    
    # 计算指标
    metrics = calculate_metrics(predictions, targets, scores)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    print(f"混淆矩阵已保存到 {cm_path}")
    
    # 绘制ROC曲线
    roc_path = os.path.join(args.results_dir, 'roc_curve.png')
    plot_roc_curve(targets, scores, roc_path)
    print(f"ROC曲线已保存到 {roc_path}")
    
    # 分析错误预测
    errors = analyze_errors(predictions, targets, scores, test_dataset, args.results_dir)
    
    # 保存评估结果
    results = {
        'test_loss': test_loss,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'num_errors': len(errors)
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(args.results_dir, 'evaluation_results.csv'), index=False)
    print(f"评估结果已保存到 {os.path.join(args.results_dir, 'evaluation_results.csv')}")
    
    print("评估完成！")

if __name__ == "__main__":
    main()