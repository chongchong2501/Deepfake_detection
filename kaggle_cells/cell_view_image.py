# Cell: Kaggle人脸提取可视化工具
"""
在Kaggle环境中可视化人脸提取效果
直接在notebook中显示提取的帧和人脸检测结果
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torch
import random
from pathlib import Path
import pandas as pd
from IPython.display import display, HTML

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_and_visualize_extracted_frames(frame_path, max_display=8):
    """
    加载并可视化已提取的帧数据
    
    Args:
        frame_path: .pt文件路径
        max_display: 最大显示帧数
    """
    try:
        # 加载预提取的帧
        frames_tensor = torch.load(frame_path, map_location='cpu')
        
        # 确保是正确的格式
        if isinstance(frames_tensor, dict):
            frames_tensor = frames_tensor['frames']
        
        # 转换为numpy格式用于显示
        if frames_tensor.dtype != torch.float32:
            frames_tensor = frames_tensor.float()
        
        # 如果值在0-1范围，转换为0-255
        if frames_tensor.max() <= 1.0:
            frames_tensor = frames_tensor * 255.0
        
        # 转换为numpy: (T, C, H, W) -> (T, H, W, C)
        frames_np = frames_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        
        # 限制显示数量
        frames_np = frames_np[:max_display]
        
        return frames_np
        
    except Exception as e:
        print(f"❌ 加载帧数据失败: {e}")
        return None

def visualize_frames_grid(frames_np, title="提取的帧", figsize=(15, 10)):
    """
    以网格形式显示帧
    """
    if frames_np is None or len(frames_np) == 0:
        print("❌ 没有可显示的帧")
        return
    
    num_frames = len(frames_np)
    cols = min(4, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        
        if i < num_frames:
            axes[row, col].imshow(frames_np[i])
            axes[row, col].set_title(f'帧 {i+1}', fontsize=10)
        else:
            axes[row, col].axis('off')
        
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    plt.tight_layout()
    plt.show()

def analyze_frame_quality(frames_np):
    """
    分析帧质量
    """
    if frames_np is None or len(frames_np) == 0:
        return
    
    print("📊 帧质量分析:")
    print(f"  总帧数: {len(frames_np)}")
    print(f"  帧尺寸: {frames_np[0].shape}")
    
    # 计算每帧的质量指标
    quality_scores = []
    brightness_scores = []
    
    for i, frame in enumerate(frames_np):
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 计算清晰度（拉普拉斯方差）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_scores.append(laplacian_var)
        
        # 计算亮度
        brightness = np.mean(gray)
        brightness_scores.append(brightness)
    
    print(f"  平均清晰度: {np.mean(quality_scores):.2f}")
    print(f"  清晰度范围: {np.min(quality_scores):.2f} - {np.max(quality_scores):.2f}")
    print(f"  平均亮度: {np.mean(brightness_scores):.2f}")
    print(f"  亮度范围: {np.min(brightness_scores):.2f} - {np.max(brightness_scores):.2f}")
    
    # 绘制质量分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(quality_scores, 'b-o', markersize=4)
    ax1.set_title('帧清晰度分布')
    ax1.set_xlabel('帧索引')
    ax1.set_ylabel('清晰度分数')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(brightness_scores, 'r-o', markersize=4)
    ax2.set_title('帧亮度分布')
    ax2.set_xlabel('帧索引')
    ax2.set_ylabel('亮度值')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_real_vs_fake_frames(data_dir='./data'):
    """
    对比真实视频和假视频的帧
    """
    frames_dir = os.path.join(data_dir, 'frames')
    
    if not os.path.exists(frames_dir):
        print(f"❌ 帧目录不存在: {frames_dir}")
        return
    
    # 查找.pt文件
    pt_files = [f for f in os.listdir(frames_dir) if f.endswith('.pt')]
    
    if not pt_files:
        print(f"❌ 在 {frames_dir} 中未找到.pt文件")
        return
    
    print(f"📁 找到 {len(pt_files)} 个帧文件")
    
    # 加载CSV文件获取标签信息
    csv_files = ['train.csv', 'val.csv', 'test.csv']
    all_data = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_data.extend(df.to_dict('records'))
    
    if not all_data:
        print("❌ 未找到CSV标签文件")
        return
    
    # 分离真实和假视频
    real_samples = [item for item in all_data if item['label'] == 0]
    fake_samples = [item for item in all_data if item['label'] == 1]
    
    print(f"📊 数据统计:")
    print(f"  真实视频样本: {len(real_samples)}")
    print(f"  假视频样本: {len(fake_samples)}")
    
    # 随机选择样本进行对比
    if real_samples and fake_samples:
        real_sample = random.choice(real_samples)
        fake_sample = random.choice(fake_samples)
        
        print(f"\n🎬 对比样本:")
        print(f"  真实视频: {os.path.basename(real_sample['frame_path'])}")
        print(f"  假视频: {os.path.basename(fake_sample['frame_path'])} ({fake_sample['method']})")
        
        # 加载并显示帧
        real_frames = load_and_visualize_extracted_frames(real_sample['frame_path'], max_display=6)
        fake_frames = load_and_visualize_extracted_frames(fake_sample['frame_path'], max_display=6)
        
        if real_frames is not None and fake_frames is not None:
            # 创建对比图
            fig = plt.figure(figsize=(18, 10))
            gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.2)
            
            fig.suptitle('真实视频 vs 假视频帧对比', fontsize=16, fontweight='bold')
            
            # 显示真实视频帧
            for i in range(min(6, len(real_frames))):
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(real_frames[i])
                ax.set_title(f'真实帧 {i+1}', fontsize=10)
                ax.axis('off')
            
            # 显示假视频帧
            for i in range(min(6, len(fake_frames))):
                ax = fig.add_subplot(gs[1, i])
                ax.imshow(fake_frames[i])
                ax.set_title(f'假视频帧 {i+1}\n({fake_sample["method"]})', fontsize=10)
                ax.axis('off')
            
            plt.show()
            
            # 分析质量差异
            print("\n📊 真实视频帧质量分析:")
            analyze_frame_quality(real_frames)
            
            print("\n📊 假视频帧质量分析:")
            analyze_frame_quality(fake_frames)

def show_sample_frames_by_method(data_dir='./data', max_samples_per_method=2):
    """
    按方法显示样本帧
    """
    # 加载数据
    csv_files = ['train.csv', 'val.csv', 'test.csv']
    all_data = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_data.extend(df.to_dict('records'))
    
    if not all_data:
        print("❌ 未找到数据文件")
        return
    
    # 按方法分组
    method_groups = {}
    for item in all_data:
        method = item['method']
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append(item)
    
    print(f"📊 找到 {len(method_groups)} 种方法:")
    for method, items in method_groups.items():
        print(f"  {method}: {len(items)} 个样本")
    
    # 为每种方法显示样本
    for method, items in method_groups.items():
        print(f"\n🎭 显示 {method} 方法的样本帧:")
        
        # 随机选择样本
        samples = random.sample(items, min(max_samples_per_method, len(items)))
        
        for i, sample in enumerate(samples):
            print(f"\n  样本 {i+1}: {os.path.basename(sample['frame_path'])}")
            frames = load_and_visualize_extracted_frames(sample['frame_path'], max_display=4)
            
            if frames is not None:
                title = f"{method} - 样本 {i+1}"
                if sample['label'] == 0:
                    title += " (真实视频)"
                else:
                    title += " (假视频)"
                
                visualize_frames_grid(frames, title=title, figsize=(12, 6))

def quick_preview_frames(data_dir='./data', num_samples=3):
    """
    快速预览提取的帧
    """
    frames_dir = os.path.join(data_dir, 'frames')
    
    if not os.path.exists(frames_dir):
        print(f"❌ 帧目录不存在: {frames_dir}")
        return
    
    # 获取所有.pt文件
    pt_files = [f for f in os.listdir(frames_dir) if f.endswith('.pt')]
    
    if not pt_files:
        print(f"❌ 在 {frames_dir} 中未找到.pt文件")
        return
    
    print(f"📁 找到 {len(pt_files)} 个帧文件")
    
    # 随机选择几个文件进行预览
    sample_files = random.sample(pt_files, min(num_samples, len(pt_files)))
    
    for i, pt_file in enumerate(sample_files):
        print(f"\n🎬 预览文件 {i+1}: {pt_file}")
        
        frame_path = os.path.join(frames_dir, pt_file)
        frames = load_and_visualize_extracted_frames(frame_path, max_display=6)
        
        if frames is not None:
            title = f"预览 {i+1}: {pt_file}"
            visualize_frames_grid(frames, title=title, figsize=(15, 8))
            
            # 简单质量分析
            print(f"  帧数: {len(frames)}")
            print(f"  尺寸: {frames[0].shape}")
            
            # 计算平均亮度
            avg_brightness = np.mean([np.mean(frame) for frame in frames])
            print(f"  平均亮度: {avg_brightness:.2f}")

def main_visualization():
    """
    主可视化函数 - 在Kaggle中运行
    """
    print("🎭 Kaggle人脸提取可视化工具")
    print("=" * 50)
    
    # 检查数据目录
    data_dir = './data'
    frames_dir = os.path.join(data_dir, 'frames')
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请先运行数据准备脚本")
        return
    
    if not os.path.exists(frames_dir):
        print(f"❌ 帧目录不存在: {frames_dir}")
        print("请先运行数据准备脚本")
        return
    
    # 统计文件
    pt_files = [f for f in os.listdir(frames_dir) if f.endswith('.pt')]
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    print(f"📁 数据统计:")
    print(f"  帧文件: {len(pt_files)} 个")
    print(f"  CSV文件: {len(csv_files)} 个")
    
    if not pt_files:
        print("❌ 未找到帧文件，请先运行数据准备")
        return
    
    # 1. 快速预览
    print("\n" + "="*30)
    print("1️⃣ 快速预览提取的帧")
    print("="*30)
    quick_preview_frames(data_dir, num_samples=3)
    
    # 2. 显示真实vs假视频对比
    print("\n" + "="*30)
    print("2️⃣ 真实 vs 假视频对比")
    print("="*30)
    compare_real_vs_fake_frames(data_dir)
    
    # 3. 按方法显示样本
    print("\n" + "="*30)
    print("3️⃣ 各方法样本展示")
    print("="*30)
    show_sample_frames_by_method(data_dir, max_samples_per_method=1)
    
    print("\n✅ 可视化完成！")
    print("💡 提示: 如果要查看更多样本，可以重新运行相关函数")

# 在Kaggle中运行
if __name__ == "__main__":
    main_visualization()