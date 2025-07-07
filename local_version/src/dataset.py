# 数据集类 - 本地RTX4070优化版本

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import config
from data_processing import VideoProcessor

class DeepfakeVideoDataset(Dataset):
    """深度伪造视频数据集类 - RTX4070优化版本"""
    
    def __init__(self, csv_file=None, data_list=None, transform=None, max_frames=None, 
                 gpu_preprocessing=True, cache_frames=True):
        if csv_file is not None:
            self.df = pd.read_csv(csv_file)
            self.data_list = None
        elif data_list is not None:
            self.data_list = data_list
            self.df = None
        else:
            raise ValueError("必须提供csv_file或data_list")
            
        self.transform = transform
        self.max_frames = max_frames or config.MAX_FRAMES
        self.gpu_preprocessing = gpu_preprocessing and torch.cuda.is_available()
        self.cache_frames = cache_frames
        self.device = config.get_device()
        
        # RTX4070优化：更大的缓存系统
        self.frame_cache = {} if cache_frames else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 视频处理器
        self.video_processor = VideoProcessor(
            max_frames=self.max_frames,
            target_size=config.FRAME_SIZE
        )
        
        # GPU预处理的标准化参数
        if self.gpu_preprocessing:
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32)
            
        print(f"🚀 数据集初始化: GPU预处理={'启用' if self.gpu_preprocessing else '禁用'}, "
              f"缓存={'启用' if self.cache_frames else '禁用'}, 设备={self.device}")
    
    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.data_list is not None:
            # 直接从内存中的数据列表获取
            item = self.data_list[idx]
            frames = item['frames']
            label = item['label']
            video_path = item.get('video_path', None)
        else:
            # 从CSV文件获取路径
            row = self.df.iloc[idx]
            video_path = row['video_path']
            label = row['label']
            frames = None
        
        # 获取视频帧
        if frames is None:
            # 检查缓存
            if self.cache_frames and video_path in self.frame_cache:
                frames = self.frame_cache[video_path]
                self.cache_hits += 1
            else:
                frames = self.video_processor.extract_frames_gpu_accelerated(video_path)
                self.cache_misses += 1
                # 缓存帧数据
                if self.cache_frames and len(frames) > 0:
                    self.frame_cache[video_path] = frames
        
        # 确保有足够的帧
        frames = self._ensure_frame_count(frames)
        
        # 数据预处理
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            video_tensor = torch.stack(frames)
        elif self.gpu_preprocessing:
            video_tensor = self._gpu_preprocess(frames)
        else:
            video_tensor = self._cpu_preprocess(frames)
        
        # 标签处理
        label_tensor = torch.tensor(label, dtype=torch.float32)
        if self.gpu_preprocessing:
            label_tensor = label_tensor.to(self.device, non_blocking=True)
        
        return video_tensor, label_tensor
    
    def _ensure_frame_count(self, frames):
        """确保帧数符合要求"""
        if len(frames) == 0:
            frames = [np.zeros((*config.FRAME_SIZE, 3), dtype=np.uint8) for _ in range(self.max_frames)]
        
        # 填充或截断到指定帧数
        while len(frames) < self.max_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((*config.FRAME_SIZE, 3), dtype=np.uint8))
        
        return frames[:self.max_frames]
    
    def _gpu_preprocess(self, frames):
        """GPU预处理"""
        # 转换为tensor
        frames_array = np.stack(frames)  # (T, H, W, C)
        video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        
        # 移动到GPU并进行预处理
        video_tensor = video_tensor.to(self.device, non_blocking=True, dtype=torch.float32) / 255.0
        
        # 标准化
        video_tensor = (video_tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
        
        return video_tensor
    
    def _cpu_preprocess(self, frames):
        """CPU预处理"""
        frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
        video_tensor = torch.stack(frames)
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        return video_tensor
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.frame_cache) if self.frame_cache else 0
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self.frame_cache:
            self.frame_cache.clear()
            print("缓存已清空")

def create_data_loaders(train_data, val_data, test_data, batch_size=None, quick_mode=False):
    """创建数据加载器"""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # 在快速模式下禁用GPU预处理和缓存以节省内存
    gpu_preprocessing = not quick_mode
    cache_frames = not quick_mode
    
    # 创建数据集 - 支持DataFrame或数据列表
    if isinstance(train_data, list):
        train_dataset = DeepfakeVideoDataset(
            csv_file=None,
            data_list=train_data,
            gpu_preprocessing=gpu_preprocessing,
            cache_frames=cache_frames
        )
    else:
        # 对于DataFrame，创建一个临时的空列表来绕过初始化检查
        train_dataset = DeepfakeVideoDataset(
            csv_file=None,
            data_list=[],  # 临时空列表
            gpu_preprocessing=gpu_preprocessing,
            cache_frames=cache_frames
        )
        train_dataset.df = train_data
        train_dataset.data_list = None  # 重置为None
    
    if isinstance(val_data, list):
        val_dataset = DeepfakeVideoDataset(
            csv_file=None,
            data_list=val_data,
            gpu_preprocessing=gpu_preprocessing,
            cache_frames=cache_frames
        )
    else:
        val_dataset = DeepfakeVideoDataset(
            csv_file=None,
            data_list=[],  # 临时空列表
            gpu_preprocessing=gpu_preprocessing,
            cache_frames=cache_frames
        )
        val_dataset.df = val_data
        val_dataset.data_list = None  # 重置为None
    
    if isinstance(test_data, list):
        test_dataset = DeepfakeVideoDataset(
            csv_file=None,
            data_list=test_data,
            gpu_preprocessing=gpu_preprocessing,
            cache_frames=cache_frames
        )
    else:
        test_dataset = DeepfakeVideoDataset(
            csv_file=None,
            data_list=[],  # 临时空列表
            gpu_preprocessing=gpu_preprocessing,
            cache_frames=cache_frames
        )
        test_dataset.df = test_data
        test_dataset.data_list = None  # 重置为None
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    
    # 在快速模式下使用更保守的设置以节省内存
    num_workers = 0 if quick_mode else config.NUM_WORKERS
    pin_memory = False if quick_mode else config.PIN_MEMORY
    prefetch_factor = None if quick_mode else config.PREFETCH_FACTOR  # num_workers=0时必须为None
    persistent_workers = False if quick_mode else config.PERSISTENT_WORKERS
    
    # 构建DataLoader参数
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers
    }
    
    # 只有在多进程模式下才设置prefetch_factor
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    print(f"\n📊 数据加载器统计:")
    print(f"训练批次数: {len(train_loader)} (批次大小: {batch_size})")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    print(f"数据加载worker数: {num_workers}")
    if quick_mode:
        print(f"⚡ 快速模式优化: GPU预处理={gpu_preprocessing}, 缓存={cache_frames}, Pin内存={pin_memory}")
    
    return train_loader, val_loader, test_loader