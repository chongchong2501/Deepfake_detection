# Cell 4: 数据集类定义

import os
import numpy as np

class DeepfakeVideoDataset(Dataset):
    """深度伪造视频数据集类 - Kaggle T4 优化版本"""
    
    def __init__(self, csv_file=None, data_list=None, transform=None, max_frames=16, 
                 gpu_preprocessing=True, cache_frames=False):
        if csv_file is not None:
            self.df = pd.read_csv(csv_file)
            self.data_list = None
        elif data_list is not None:
            self.data_list = data_list
            self.df = None
        else:
            raise ValueError("必须提供csv_file或data_list")
            
        self.transform = transform
        self.max_frames = max_frames
        self.gpu_preprocessing = gpu_preprocessing and torch.cuda.is_available()
        self.cache_frames = cache_frames
        
        # 简化缓存系统 - 仅CPU缓存，避免GPU内存压力
        self.frame_cache = {} if cache_frames else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # GPU预处理的标准化参数 - 统一使用FP32
        if self.gpu_preprocessing:
            self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=torch.float32)
            self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda', dtype=torch.float32)
            
        print(f"🚀 数据集初始化: GPU预处理={'启用' if self.gpu_preprocessing else '禁用'}, "
              f"缓存={'启用' if self.cache_frames else '禁用'}, 数据类型=FP32")
        self.frame_dir = './frames'
        os.makedirs(self.frame_dir, exist_ok=True)
    
    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.data_list is not None:
            item = self.data_list[idx]
            video_path = item['video_path']
            frames = item['frames']
            label = item['label']
        else:
            row = self.df.iloc[idx]
            video_path = row['video_path']
            label = row['label']
            frames = None
        
        # 简化的数据处理流程
        npy_path = os.path.join(self.frame_dir, os.path.basename(video_path) + '.npy')
        if os.path.exists(npy_path):
            loaded_frames = np.load(npy_path)
            frames = [loaded_frames[i] for i in range(loaded_frames.shape[0])]
        else:
            if frames is None:
                # 检查CPU缓存
                if self.cache_frames and video_path in self.frame_cache:
                    frames = self.frame_cache[video_path]
                    self.cache_hits += 1
                else:
                    frames = extract_frames_gpu_accelerated(video_path, self.max_frames, target_size=(224, 224))
                    self.cache_misses += 1
                    # 缓存帧数据
                    if self.cache_frames and len(frames) > 0:
                        self.frame_cache[video_path] = frames
            # 保存预处理帧
            if len(frames) > 0:
                np.save(npy_path, np.stack(frames))
        
        # 确保有足够的帧
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_frames)]
        
        while len(frames) < self.max_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        frames = frames[:self.max_frames]
        
        # 统一的数据预处理 - 全部使用FP32
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            video_tensor = torch.stack(frames)
        elif self.gpu_preprocessing:
            # GPU预处理：减少CPU-GPU传输次数
            frames_array = np.stack(frames)  # (T, H, W, C)
            video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
            
            # 移动到GPU并进行预处理 - 统一使用FP32
            video_tensor = video_tensor.to('cuda', non_blocking=True, dtype=torch.float32) / 255.0
            
            # 标准化
            video_tensor = (video_tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
        else:
            # CPU预处理
            frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
            video_tensor = torch.stack(frames)
        
        # 标签处理 - 统一使用FP32
        label_tensor = torch.tensor(label, dtype=torch.float32)
        if self.gpu_preprocessing:
            label_tensor = label_tensor.to('cuda', non_blocking=True)
        
        return video_tensor, label_tensor
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cpu_cache_size': len(self.frame_cache) if self.frame_cache else 0
        }

print("✅ 数据集类定义完成")