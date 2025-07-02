import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_preprocessing import DeepfakeVideoDataset, set_seed
from model import create_model

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='深度伪造检测模型训练')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--model_dir', type=str, default='./models', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lightweight'], help='模型类型')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--num_frames', type=int, default=30, help='每个视频的帧数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default='', help='检查点路径')
    return parser.parse_args()

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(train_loader, desc='训练')
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if isinstance(model.forward(inputs), tuple):
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
        
        outputs = outputs.squeeze()
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        
        # 收集预测和目标
        preds = (outputs > 0.5).float().cpu().numpy()
        predictions.extend(preds)
        targets.extend(labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
    
    # 计算平均损失和指标
    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='验证')
        
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
            predictions.extend(preds)
            targets.extend(labels.cpu().numpy())
            scores.extend(outputs.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
    
    # 计算平均损失和指标
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    auc = roc_auc_score(targets, scores) if len(set(targets)) > 1 else 0.0
    
    return epoch_loss, accuracy, precision, recall, f1, auc

# 保存检查点
def save_checkpoint(model, optimizer, epoch, metrics, args, is_best=False):
    os.makedirs(args.model_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # 保存最新检查点
    checkpoint_path = os.path.join(args.model_dir, f'{args.model_type}_checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 如果是最佳模型，也保存一份
    if is_best:
        best_model_path = os.path.join(args.model_dir, f'{args.model_type}_model_best.pth')
        torch.save(checkpoint, best_model_path)
        print(f"保存最佳模型到 {best_model_path}")

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, args):
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, 'loss_curve.png'))
    plt.close()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot([m['accuracy'] for m in train_metrics], label='训练准确率')
    plt.plot([m['accuracy'] for m in val_metrics], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, 'accuracy_curve.png'))
    plt.close()
    
    # 绘制F1分数曲线
    plt.figure(figsize=(10, 5))
    plt.plot([m['f1'] for m in train_metrics], label='训练F1分数')
    plt.plot([m['f1'] for m in val_metrics], label='验证F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('训练和验证F1分数')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, 'f1_curve.png'))
    plt.close()

# 主函数
def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
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
    
    # 加载数据集
    train_dataset = DeepfakeVideoDataset(
        csv_file=os.path.join(args.data_dir, 'train.csv'),
        transform=transform,
        max_frames=args.num_frames
    )
    
    val_dataset = DeepfakeVideoDataset(
        csv_file=os.path.join(args.data_dir, 'val.csv'),
        transform=transform,
        max_frames=args.num_frames
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = create_model(model_type=args.model_type, device=device)
    print(f"模型类型: {args.model_type}")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 初始化变量
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    # 如果从检查点恢复训练
    if args.resume and args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"加载检查点 '{args.checkpoint}'")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['metrics']['val_loss']
            print(f"从epoch {start_epoch}继续训练")
        else:
            print(f"未找到检查点 '{args.checkpoint}'")
    
    # 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # 训练
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_metrics.append({
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1
        })
        
        val_metrics.append({
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_rec,
            'f1': val_f1,
            'auc': val_auc
        })
        
        # 打印指标
        print(f"训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # 检查是否是最佳模型
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # 保存检查点
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'val_auc': val_auc
        }
        save_checkpoint(model, optimizer, epoch, metrics, args, is_best)
        
        # 绘制训练曲线
        plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, args)
    
    print("训练完成！")

if __name__ == "__main__":
    main()