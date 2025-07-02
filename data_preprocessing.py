import os
import cv2
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import zipfile
import shutil
import kaggle
from tqdm import tqdm

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 下载FaceForensics++数据集
def download_dataset():
    print("正在下载FaceForensics++数据集...")
    # 使用Kaggle API下载数据集
    # 注意：需要先设置Kaggle API凭证
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('c23/faceforensics', path='./data', unzip=True)
    print("数据集下载完成！")

# 提取视频帧
def extract_frames(video_path, output_dir, frames_per_video=30, resize_dim=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 均匀采样帧
    if frame_count <= frames_per_video:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = np.linspace(0, frame_count-1, frames_per_video, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 调整大小
            frame = cv2.resize(frame, resize_dim)
            # 保存帧
            output_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_frame_{i:03d}.jpg")
            cv2.imwrite(output_path, frame)
    
    cap.release()

# 处理所有视频
def process_videos(data_dir, output_dir, frames_per_video=30):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理真实视频
    real_dir = os.path.join(data_dir, 'original_sequences', 'c23', 'videos')
    real_output_dir = os.path.join(output_dir, 'real')
    os.makedirs(real_output_dir, exist_ok=True)
    
    print("处理真实视频...")
    for video_file in tqdm(os.listdir(real_dir)):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(real_dir, video_file)
            video_output_dir = os.path.join(real_output_dir, video_file.split('.')[0])
            extract_frames(video_path, video_output_dir, frames_per_video)
    
    # 处理伪造视频 - Deepfakes
    fake_dir = os.path.join(data_dir, 'manipulated_sequences', 'Deepfakes', 'c23', 'videos')
    fake_output_dir = os.path.join(output_dir, 'fake')
    os.makedirs(fake_output_dir, exist_ok=True)
    
    print("处理伪造视频...")
    for video_file in tqdm(os.listdir(fake_dir)):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(fake_dir, video_file)
            video_output_dir = os.path.join(fake_output_dir, video_file.split('.')[0])
            extract_frames(video_path, video_output_dir, frames_per_video)

# 创建数据集CSV文件
def create_dataset_csv(frames_dir, output_csv):
    data = []
    
    # 处理真实视频帧
    real_dir = os.path.join(frames_dir, 'real')
    for video_dir in os.listdir(real_dir):
        video_frames_dir = os.path.join(real_dir, video_dir)
        if os.path.isdir(video_frames_dir):
            frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')]
            frame_files.sort()
            
            # 将帧路径和标签添加到数据列表
            video_data = {
                'video_id': video_dir,
                'frames': [os.path.join(video_frames_dir, f) for f in frame_files],
                'label': 0  # 0表示真实
            }
            data.append(video_data)
    
    # 处理伪造视频帧
    fake_dir = os.path.join(frames_dir, 'fake')
    for video_dir in os.listdir(fake_dir):
        video_frames_dir = os.path.join(fake_dir, video_dir)
        if os.path.isdir(video_frames_dir):
            frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')]
            frame_files.sort()
            
            # 将帧路径和标签添加到数据列表
            video_data = {
                'video_id': video_dir,
                'frames': [os.path.join(video_frames_dir, f) for f in frame_files],
                'label': 1  # 1表示伪造
            }
            data.append(video_data)
    
    # 保存为CSV文件
    df = pd.DataFrame({
        'video_id': [item['video_id'] for item in data],
        'frames': [item['frames'] for item in data],
        'label': [item['label'] for item in data]
    })
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # 保存训练集和验证集
    train_df.to_csv(os.path.join(output_csv, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_csv, 'val.csv'), index=False)
    
    print(f"数据集CSV文件已创建：{output_csv}")
    print(f"训练集大小：{len(train_df)}，验证集大小：{len(val_df)}")

# 视频数据集类
class DeepfakeVideoDataset(Dataset):
    def __init__(self, csv_file, transform=None, max_frames=30):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_data = self.data.iloc[idx]
        frames_paths = eval(video_data['frames'])  # 将字符串转换回列表
        label = video_data['label']
        
        # 加载帧
        frames = []
        for frame_path in frames_paths[:self.max_frames]:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB
            
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        # 如果帧数不足，用零填充
        if len(frames) < self.max_frames:
            zero_frame = torch.zeros_like(frames[0])
            frames.extend([zero_frame] * (self.max_frames - len(frames)))
        
        # 将帧堆叠成张量
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, label

# 主函数
def main():
    set_seed(42)
    
    # 设置路径
    data_dir = './data/faceforensics'
    frames_dir = './data/processed_frames'
    csv_dir = './data'
    
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # 下载数据集
    download_dataset()
    
    # 处理视频
    process_videos(data_dir, frames_dir)
    
    # 创建数据集CSV文件
    create_dataset_csv(frames_dir, csv_dir)
    
    print("数据预处理完成！")

if __name__ == "__main__":
    main()