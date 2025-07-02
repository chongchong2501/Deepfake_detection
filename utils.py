import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    """设置随机种子以确保实验可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 创建检查点目录
def create_checkpoint_dir(checkpoint_dir):
    """创建检查点目录
    
    Args:
        checkpoint_dir: 检查点目录路径
    
    Returns:
        checkpoint_dir: 创建的检查点目录路径
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

# 保存检查点
def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path):
    """保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        metrics: 评估指标
        checkpoint_path: 检查点保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存到 {checkpoint_path}")

# 加载检查点
def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        device: 设备
    
    Returns:
        model: 加载检查点后的模型
        optimizer: 加载检查点后的优化器（如果提供）
        epoch: 检查点的轮次
        metrics: 检查点的评估指标
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    print(f"从 {checkpoint_path} 加载检查点，轮次 {epoch}")
    
    return model, optimizer, epoch, metrics

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_dir):
    """绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_metrics: 训练指标字典列表
        val_metrics: 验证指标字典列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot([m['accuracy'] for m in train_metrics], label='训练准确率')
    plt.plot([m['accuracy'] for m in val_metrics], label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()
    
    # 绘制F1分数曲线
    plt.figure(figsize=(10, 5))
    plt.plot([m['f1'] for m in train_metrics], label='训练F1分数')
    plt.plot([m['f1'] for m in val_metrics], label='验证F1分数')
    plt.xlabel('轮次')
    plt.ylabel('F1分数')
    plt.title('训练和验证F1分数')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'f1_curve.png'))
    plt.close()

# 绘制ROC曲线
def plot_roc_curve(y_true, y_score, save_path=None):
    """绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        save_path: 保存路径（可选）
    
    Returns:
        auc_score: AUC分数
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return roc_auc

# 提取视频帧
def extract_video_frames(video_path, output_dir, num_frames=30, resize_dim=(128, 128)):
    """从视频中提取帧
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        num_frames: 提取的帧数
        resize_dim: 调整大小的尺寸
    
    Returns:
        frame_paths: 提取的帧路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 均匀采样帧
    if frame_count <= num_frames:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
    
    frame_paths = []
    
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="提取帧")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 调整大小
            frame = cv2.resize(frame, resize_dim)
            # 保存帧
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    
    cap.release()
    
    return frame_paths