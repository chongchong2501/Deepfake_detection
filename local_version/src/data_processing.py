# 数据处理模块 - 本地RTX4070优化版本

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import read_video
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from config import config

# 检查PyAV可用性
try:
    import av
    PYAV_AVAILABLE = True
    print("✅ PyAV已安装，支持GPU视频处理")
except ImportError:
    PYAV_AVAILABLE = False
    print("⚠️ PyAV未安装，视频处理将回退到CPU模式")

class VideoProcessor:
    """视频处理类 - RTX4070优化"""
    
    def __init__(self, max_frames=16, target_size=(224, 224), quality_threshold=20):
        self.max_frames = max_frames
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        self.device = config.get_device()
    
    def extract_frames_gpu_accelerated(self, video_path, use_gpu=True):
        """GPU加速的帧提取函数"""
        try:
            # 检查PyAV是否可用
            if not PYAV_AVAILABLE:
                print(f"PyAV不可用，使用CPU回退处理: {video_path}")
                return self.extract_frames_cpu_fallback(video_path)
                
            # 使用torchvision的GPU加速视频读取
            if use_gpu and torch.cuda.is_available():
                device = self.device
            else:
                device = torch.device('cpu')
                
            # 读取视频（torchvision自动处理解码）
            try:
                video_tensor, audio, info = read_video(video_path, pts_unit='sec')
                # video_tensor shape: (T, H, W, C)
            except Exception as e:
                print(f"GPU视频读取失败，回退到CPU: {e}")
                return self.extract_frames_cpu_fallback(video_path)
            
            if video_tensor.size(0) == 0:
                return []
                
            # 移动到GPU进行处理
            video_tensor = video_tensor.to(device, non_blocking=True)
            total_frames = video_tensor.size(0)
            
            # 智能帧采样策略
            if total_frames <= self.max_frames:
                frame_indices = torch.arange(0, total_frames, device=device)
            else:
                # 均匀采样
                step = total_frames / self.max_frames
                frame_indices = torch.arange(0, total_frames, step, device=device).long()[:self.max_frames]
            
            # 批量提取帧
            selected_frames = video_tensor[frame_indices]  # (max_frames, H, W, C)
            
            # GPU上进行质量检测（使用Sobel算子代替Laplacian）
            if self.quality_threshold > 0:
                # 转换为灰度图进行质量检测（先转换为float类型）
                gray_frames = selected_frames.float().mean(dim=-1, keepdim=True)  # (T, H, W, 1)
                gray_frames = gray_frames.permute(0, 3, 1, 2)  # (T, 1, H, W)
                
                # 使用Sobel算子计算图像质量
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32, device=device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32, device=device).view(1, 1, 3, 3)
                
                grad_x = F.conv2d(gray_frames, sobel_x, padding=1)
                grad_y = F.conv2d(gray_frames, sobel_y, padding=1)
                quality_scores = (grad_x.pow(2) + grad_y.pow(2)).mean(dim=[1, 2, 3])
                
                # 过滤低质量帧
                quality_mask = quality_scores > self.quality_threshold
                if quality_mask.sum() > 0:
                    selected_frames = selected_frames[quality_mask]
                
            # GPU上进行尺寸调整
            selected_frames = selected_frames.permute(0, 3, 1, 2).float()  # (T, C, H, W)
            if selected_frames.size(-1) != self.target_size[0] or selected_frames.size(-2) != self.target_size[1]:
                selected_frames = F.interpolate(selected_frames, size=self.target_size, 
                                              mode='bilinear', align_corners=False)
            
            # 确保帧数足够
            current_frames = selected_frames.size(0)
            if current_frames < self.max_frames:
                # 重复最后一帧
                if current_frames > 0:
                    last_frame = selected_frames[-1:].repeat(self.max_frames - current_frames, 1, 1, 1)
                    selected_frames = torch.cat([selected_frames, last_frame], dim=0)
                else:
                    # 创建黑色帧
                    selected_frames = torch.zeros(self.max_frames, 3, self.target_size[0], self.target_size[1], 
                                                device=device, dtype=torch.float32)
            
            # 限制到最大帧数
            selected_frames = selected_frames[:self.max_frames]
            
            # 转换回CPU numpy格式（为了兼容现有代码）
            frames_cpu = selected_frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            frames_list = [frame for frame in frames_cpu]
            
            return frames_list
            
        except Exception as e:
            print(f"GPU帧提取失败，回退到CPU: {e}")
            return self.extract_frames_cpu_fallback(video_path)
    
    def extract_frames_cpu_fallback(self, video_path):
        """CPU回退的帧提取函数"""
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
        if total_frames <= self.max_frames:
            frame_indices = list(range(0, total_frames, max(1, total_frames // self.max_frames)))
        else:
            step = max(1, total_frames // self.max_frames)
            frame_indices = list(range(0, total_frames, step))[:self.max_frames]

        frame_count = 0
        for frame_idx in frame_indices:
            if frame_count >= self.max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 质量检测
                if self.quality_threshold > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    quality = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if quality <= self.quality_threshold:
                        continue
                
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
                frame_count += 1

        cap.release()

        # 如果帧数不足，重复最后一帧
        while len(frames) < self.max_frames and len(frames) > 0:
            frames.append(frames[-1].copy())

        return frames[:self.max_frames]
    
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
                    frames = self.extract_frames_gpu_accelerated(video_path)
                    
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
                        frames = self.extract_frames_gpu_accelerated(video_path)
                        
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
    processor = VideoProcessor(
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