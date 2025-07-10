# 数据集类 - 本地RTX4070优化版本

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import config
from video_processor import get_video_processor, process_video_safe, read_video_safe
from memory_manager import auto_memory_management, cleanup_memory

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
        self.video_processor = get_video_processor()
        
        # GPU预处理的标准化参数
        if self.gpu_preprocessing:
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32)
    
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
                # 使用新的安全视频读取
                raw_frames = read_video_safe(video_path, max_frames=self.max_frames)
                if not raw_frames:
                    # 如果读取失败，返回默认帧
                    frames = []
                else:
                    frames = process_video_safe(raw_frames)
                
                self.cache_misses += 1
                # 缓存帧数据
                if self.cache_frames and len(frames) > 0 and len(self.frame_cache) < 1000:  # 限制缓存大小
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
    
    @auto_memory_management(cleanup_interval=100)
    def _gpu_preprocess(self, frames):
        """GPU预处理（带内存管理）"""
        try:
            # 确保frames是numpy数组
            if isinstance(frames[0], torch.Tensor):
                # 如果是tensor，先移动到CPU再转换为numpy
                frames = [frame.cpu().numpy() if frame.is_cuda else frame.numpy() for frame in frames]
            
            # 转换为tensor
            frames_array = np.stack(frames)  # (T, H, W, C)
            video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
            
            # 移动到GPU并进行预处理
            video_tensor = video_tensor.to(self.device, non_blocking=True, dtype=torch.float32) / 255.0
            
            # 标准化
            video_tensor = (video_tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
            
            return video_tensor
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"🚨 GPU内存不足，清理缓存后重试...")
                cleanup_memory(force=True)
                self.clear_cache()
                # 重试一次
                try:
                    frames_array = np.stack(frames)
                    video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()
                    video_tensor = video_tensor.to(self.device, non_blocking=True, dtype=torch.float32) / 255.0
                    video_tensor = (video_tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
                    return video_tensor
                except:
                    print(f"GPU预处理重试失败，回退到CPU")
                    return self._cpu_preprocess(frames)
            else:
                print(f"GPU预处理失败: {e}，回退到CPU")
                return self._cpu_preprocess(frames)
    
    def _cpu_preprocess(self, frames):
        """CPU预处理"""
        import cv2
        
        # 确保frames是numpy数组
        if isinstance(frames[0], torch.Tensor):
            frames = [frame.cpu().numpy() if frame.is_cuda else frame.numpy() for frame in frames]
        
        # 调整所有帧到相同尺寸 (224, 224)
        processed_frames = []
        for frame in frames:
            # 确保frame是正确的形状 (H, W, C)
            if frame.shape[-1] != 3:
                frame = frame.transpose(1, 2, 0)  # 从 (C, H, W) 转换为 (H, W, C)
            
            # 调整尺寸到224x224
            frame_resized = cv2.resize(frame, (224, 224))
            processed_frames.append(frame_resized)
        
        # 转换为tensor
        frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in processed_frames]
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
            cache_size_before = len(self.frame_cache)
            self.frame_cache.clear()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 清理GPU内存
            cleanup_memory()
            
            print(f"🧹 缓存已清空: 释放了{cache_size_before}个缓存项")

def create_data_loaders(train_data, val_data, test_data, batch_size=None, quick_mode=False):
    """创建数据加载器（带内存管理）"""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # 内存优化：根据可用内存调整参数
    try:
        from memory_manager import get_memory_manager
        memory_manager = get_memory_manager()
        stats = memory_manager.get_memory_stats()
        
        # 根据内存使用情况自动调整参数
        if stats.gpu_memory_percent > 0.7:
            print("⚠️ GPU内存使用率较高，自动优化参数")
            if batch_size > 4:
                batch_size = max(4, batch_size // 2)
                print(f"   batch_size调整为: {batch_size}")
            quick_mode = True
        
        if stats.cpu_memory_percent > 0.8:
            print("⚠️ CPU内存使用率较高，自动优化参数")
            quick_mode = True
    except ImportError:
        print("内存管理器不可用，使用默认设置")
    
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
    persistent_workers = False if (quick_mode or num_workers == 0) else config.PERSISTENT_WORKERS
    
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