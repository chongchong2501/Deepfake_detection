#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 第2段：轻量级优化的数据下载和预处理（Kaggle友好版本）
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

# =============================================================================
# 第2段：轻量级优化的数据下载和预处理（Kaggle友好版本）
# =============================================================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')
import gc
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# 检查是否在Kaggle环境中
IS_KAGGLE = os.path.exists('/kaggle')

if IS_KAGGLE:
    BASE_DATA_DIR = '/kaggle/input/ff-c23/FaceForensics++_C23'
    print("检测到Kaggle环境")
    print(f"数据基础路径: {BASE_DATA_DIR}")
else:
    BASE_DATA_DIR = './FaceForensics++_C23'
    print("本地环境")

# 内存友好的帧提取函数
def extract_frames_memory_efficient(video_path, max_frames=24, target_size=(160, 160), 
                                   quality_threshold=30, skip_frames=2):
    """
    内存友好的帧提取函数
    - 降低分辨率减少内存使用
    - 减少帧数
    - 添加跳帧机制
    - 简化质量检测
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return frames
    
    # 简化采样策略：均匀采样，避免复杂计算
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
            # 转换颜色空间
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 简化质量检测（使用更快的方法）
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            quality = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if quality > quality_threshold:
                # 调整大小
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
                frame_count += 1
    
    cap.release()
    
    # 如果帧数不足，简单重复最后一帧
    while len(frames) < max_frames and len(frames) > 0:
        frames.append(frames[-1].copy())
    
    return frames[:max_frames]

# 简化的视频处理函数
def process_videos_simple(base_data_dir, max_videos_per_class=80, max_frames=24):
    """
    简化的视频处理函数，避免并发和复杂操作
    """
    data_list = []
    
    # 定义类别映射
    fake_methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    
    print("开始处理真实视频...")
    # 处理真实视频
    original_dir = os.path.join(base_data_dir, 'original')
    if os.path.exists(original_dir):
        video_files = [f for f in os.listdir(original_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # 限制视频数量
        if len(video_files) > max_videos_per_class:
            video_files = random.sample(video_files, max_videos_per_class)
        
        print(f"找到 {len(video_files)} 个真实视频")
        
        for i, video_file in enumerate(tqdm(video_files, desc="处理真实视频")):
            try:
                video_path = os.path.join(original_dir, video_file)
                
                # 提取帧
                frames = extract_frames_memory_efficient(video_path, max_frames)
                
                if len(frames) > 0:
                    # 保存帧数据路径
                    frame_save_dir = os.path.join('./data', 'frames', 'real')
                    os.makedirs(frame_save_dir, exist_ok=True)
                    
                    video_name = os.path.splitext(video_file)[0]
                    frame_save_path = os.path.join(frame_save_dir, f"{video_name}.npy")
                    
                    # 保存帧数据
                    np.save(frame_save_path, np.array(frames, dtype=np.uint8))
                    
                    data_list.append({
                        'video_path': video_path,
                        'frame_path': frame_save_path,
                        'label': 0,
                        'category': 'real',
                        'method': 'original',
                        'num_frames': len(frames),
                        'video_name': video_name
                    })
                
                # 每处理10个视频清理一次内存
                if (i + 1) % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {e}")
                continue
    
    print("开始处理伪造视频...")
    # 处理伪造视频
    for method in fake_methods:
        method_dir = os.path.join(base_data_dir, method)
        if os.path.exists(method_dir):
            video_files = [f for f in os.listdir(method_dir) 
                          if f.endswith(('.mp4', '.avi', '.mov'))]
            
            # 限制每种方法的视频数量
            method_limit = max_videos_per_class // len(fake_methods)
            if len(video_files) > method_limit:
                video_files = random.sample(video_files, method_limit)
            
            print(f"处理 {method}: {len(video_files)} 个视频")
            
            for i, video_file in enumerate(tqdm(video_files, desc=f"处理{method}")):
                try:
                    video_path = os.path.join(method_dir, video_file)
                    
                    # 提取帧
                    frames = extract_frames_memory_efficient(video_path, max_frames)
                    
                    if len(frames) > 0:
                        # 保存帧数据路径
                        frame_save_dir = os.path.join('./data', 'frames', 'fake')
                        os.makedirs(frame_save_dir, exist_ok=True)
                        
                        video_name = os.path.splitext(video_file)[0]
                        frame_save_path = os.path.join(frame_save_dir, f"{method}_{video_name}.npy")
                        
                        # 保存帧数据
                        np.save(frame_save_path, np.array(frames, dtype=np.uint8))
                        
                        data_list.append({
                            'video_path': video_path,
                            'frame_path': frame_save_path,
                            'label': 1,
                            'category': 'fake',
                            'method': method,
                            'num_frames': len(frames),
                            'video_name': video_name
                        })
                    
                    # 每处理5个视频清理一次内存
                    if (i + 1) % 5 == 0:
                        gc.collect()
                        
                except Exception as e:
                    print(f"处理视频 {video_file} 时出错: {e}")
                    continue
    
    print(f"总共成功处理了 {len(data_list)} 个视频")
    print(f"真实视频: {len([d for d in data_list if d['label'] == 0])} 个")
    print(f"伪造视频: {len([d for d in data_list if d['label'] == 1])} 个")
    
    return data_list

# 简化的数据集划分
def create_simple_dataset_split(data_list, test_size=0.2, val_size=0.1):
    """
    简化的数据集划分，避免复杂的分层采样
    """
    df = pd.DataFrame(data_list)
    
    # 简单的分层划分
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, 
        stratify=df['label'] if len(df) > 10 else None
    )
    
    if val_size > 0 and len(train_df) > 10:
        train_df, val_df = train_test_split(
            train_df, test_size=val_size/(1-test_size), random_state=42,
            stratify=train_df['label'] if len(train_df) > 10 else None
        )
        return train_df, val_df, test_df
    
    return train_df, test_df

# 数据质量检查
def check_data_quality(data_list):
    """
    简单的数据质量检查
    """
    if not data_list:
        print("❌ 没有有效的数据")
        return False
    
    df = pd.DataFrame(data_list)
    
    print("\n=== 数据统计 ===")
    print(f"总样本数: {len(df)}")
    print(f"真实视频: {len(df[df['label']==0])} 个")
    print(f"伪造视频: {len(df[df['label']==1])} 个")
    
    print("\n各方法分布:")
    method_counts = df['method'].value_counts()
    for method, count in method_counts.items():
        print(f"  {method}: {count} 个")
    
    # 检查数据平衡性
    real_count = len(df[df['label']==0])
    fake_count = len(df[df['label']==1])
    
    if real_count == 0 or fake_count == 0:
        print("⚠️ 数据严重不平衡，缺少某一类别")
        return False
    
    ratio = min(real_count, fake_count) / max(real_count, fake_count)
    if ratio < 0.3:
        print(f"⚠️ 数据不平衡，比例: {ratio:.2f}")
    else:
        print(f"✅ 数据平衡性良好，比例: {ratio:.2f}")
    
    return True

# 主处理流程
print("=== 开始数据预处理 ===")

# 检查数据目录
if IS_KAGGLE and os.path.exists(BASE_DATA_DIR):
    print("检查数据目录结构...")
    subdirs = [d for d in os.listdir(BASE_DATA_DIR) 
              if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
    print(f"找到子目录: {subdirs}")
    
    for subdir in subdirs[:6]:  # 只显示前6个，避免输出过多
        subdir_path = os.path.join(BASE_DATA_DIR, subdir)
        try:
            video_files = [f for f in os.listdir(subdir_path) 
                          if f.endswith(('.mp4', '.avi', '.mov'))]
            print(f"  {subdir}: {len(video_files)} 个视频文件")
        except:
            print(f"  {subdir}: 无法访问")

# 检查是否已有处理好的数据
if (os.path.exists('./data/train.csv') and 
    os.path.exists('./data/val.csv') and 
    os.path.exists('./data/test.csv')):
    
    print("✅ 发现已有预处理数据")
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/val.csv')
    test_df = pd.read_csv('./data/test.csv')
    
    print(f"训练集: {len(train_df)} 个样本")
    print(f"验证集: {len(val_df)} 个样本")
    print(f"测试集: {len(test_df)} 个样本")
else:
    print("开始处理视频数据...")
    
    try:
        # 处理视频
        data_list = process_videos_simple(
            BASE_DATA_DIR, 
            max_videos_per_class=200,  # 视频数量
            max_frames=30  # 帧数
        )
        
        # 检查数据质量
        if not check_data_quality(data_list):
            print("❌ 数据质量检查失败")
        else:
            # 创建数据集划分
            print("\n创建数据集划分...")
            train_df, val_df, test_df = create_simple_dataset_split(
                data_list, test_size=0.15, val_size=0.15
            )
            
            # 保存数据集
            print("保存数据集文件...")
            train_df.to_csv('./data/train.csv', index=False)
            val_df.to_csv('./data/val.csv', index=False)
            test_df.to_csv('./data/test.csv', index=False)
            
            # 保存处理信息
            process_info = {
                'total_samples': len(data_list),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'processed_time': pd.Timestamp.now().isoformat(),
                'max_frames': 20,
                'target_size': [160, 160]
            }
            
            with open('./data/process_info.json', 'w') as f:
                json.dump(process_info, f, indent=2)
            
            print(f"\n✅ 数据处理完成")
            print(f"训练集: {len(train_df)} 个样本")
            print(f"验证集: {len(val_df)} 个样本")
            print(f"测试集: {len(test_df)} 个样本")
            
            # 最终内存清理
            del data_list
            gc.collect()
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        print("建议检查数据路径和可用内存")