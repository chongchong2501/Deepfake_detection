# 数据处理模块 - 数据集管理功能

import os
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import random
from config import config

# 导入统一的视频处理器
from video_processor import get_video_processor

class DatasetManager:
    """数据集管理器 - 负责数据集的创建、划分和管理"""
    
    def __init__(self, max_frames=16, target_size=(224, 224), quality_threshold=20):
        self.max_frames = max_frames
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        # 获取优化的视频处理器
        self.video_processor = get_video_processor(
            max_frames=max_frames,
            target_size=target_size,
            quality_threshold=quality_threshold
        )
    
    def process_videos(self, base_data_dir, max_videos_per_class=250):
        """处理视频数据 - RTX4070优化版本"""
        data_list = []
        fake_methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']

        print("开始处理真实视频...")
        # 处理真实视频
        original_dir = os.path.join(base_data_dir, 'original')
        if os.path.exists(original_dir):
            video_files = [f for f in os.listdir(original_dir)
                          if f.endswith(('.mp4', '.avi', '.mov'))]
            
            if max_videos_per_class is not None and len(video_files) > max_videos_per_class:
                video_files = random.sample(video_files, max_videos_per_class)

            print(f"找到 {len(video_files)} 个真实视频")

            for video_file in tqdm(video_files, desc="处理真实视频"):
                try:
                    video_path = os.path.join(original_dir, video_file)
                    frames = self.video_processor.extract_frames_gpu_accelerated(video_path)
                    
                    if len(frames) >= self.max_frames // 2:  # 至少要有一半的帧
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
                
                if max_videos_per_class is not None and len(video_files) > max_videos_per_class:
                    video_files = random.sample(video_files, max_videos_per_class)

                print(f"处理 {method}: {len(video_files)} 个视频")

                for video_file in tqdm(video_files, desc=f"处理{method}"):
                    try:
                        video_path = os.path.join(method_dir, video_file)
                        frames = self.video_processor.extract_frames_gpu_accelerated(video_path)
                        
                        if len(frames) >= self.max_frames // 2:
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
    
    def create_dataset_split(self, data_list):
        """创建数据集划分"""
        # 分离真实和伪造数据
        real_data = [item for item in data_list if item['label'] == 0]
        fake_data = [item for item in data_list if item['label'] == 1]
        
        print(f"真实视频: {len(real_data)} 个")
        print(f"伪造视频: {len(fake_data)} 个")
        
        # 分别划分真实和伪造数据
        test_val_size = config.TEST_RATIO + config.VAL_RATIO
        test_ratio_in_temp = config.TEST_RATIO / test_val_size
        
        real_train, real_temp = train_test_split(real_data, test_size=test_val_size, random_state=config.RANDOM_SEED)
        real_val, real_test = train_test_split(real_temp, test_size=test_ratio_in_temp, random_state=config.RANDOM_SEED)
        
        fake_train, fake_temp = train_test_split(fake_data, test_size=test_val_size, random_state=config.RANDOM_SEED)
        fake_val, fake_test = train_test_split(fake_temp, test_size=test_ratio_in_temp, random_state=config.RANDOM_SEED)
        
        # 合并数据
        train_data = real_train + fake_train
        val_data = real_val + fake_val
        test_data = real_test + fake_test
        
        # 打乱数据
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        print(f"训练集: {len(train_data)} 个视频")
        print(f"验证集: {len(val_data)} 个视频")
        print(f"测试集: {len(test_data)} 个视频")
        
        return train_data, val_data, test_data
    
    def save_dataset_to_csv(self, data_list, filename):
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

def prepare_data(data_dir=None, max_videos_per_class=None, force_reprocess=False):
    """数据准备主函数"""
    if data_dir is None:
        data_dir = config.DATA_ROOT
    
    if max_videos_per_class is None:
        max_videos_per_class = config.MAX_VIDEOS_PER_CLASS
    
    # 检查是否已有处理好的数据
    train_csv = config.DATA_CACHE_DIR / "train.csv"
    val_csv = config.DATA_CACHE_DIR / "val.csv"
    test_csv = config.DATA_CACHE_DIR / "test.csv"
    
    if not force_reprocess and all(f.exists() for f in [train_csv, val_csv, test_csv]):
        print("发现已处理的数据文件，直接加载...")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        
        print(f"训练集: {len(train_df)} 个样本")
        print(f"验证集: {len(val_df)} 个样本")
        print(f"测试集: {len(test_df)} 个样本")
        
        return train_df, val_df, test_df
    
    # 创建数据目录
    config.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理视频数据
    processor = DatasetManager(
        max_frames=config.MAX_FRAMES,
        target_size=config.FRAME_SIZE
    )
    
    print(f"开始处理数据，数据路径: {data_dir}")
    data_list = processor.process_videos(data_dir, max_videos_per_class)
    
    # 划分数据集
    train_data, val_data, test_data = processor.create_dataset_split(data_list)
    
    # 保存数据集
    train_df = processor.save_dataset_to_csv(train_data, train_csv)
    val_df = processor.save_dataset_to_csv(val_data, val_csv)
    test_df = processor.save_dataset_to_csv(test_data, test_csv)
    
    return train_df, val_df, test_df