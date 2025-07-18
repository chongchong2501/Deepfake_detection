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
        
        # 优化缓存系统 - 使用LRU缓存
        if cache_frames:
            from functools import lru_cache
            self.frame_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
            self.max_cache_size = 100  # 限制缓存大小
        else:
            self.frame_cache = None
        
        # 预计算数据统计信息
        self._compute_dataset_stats()
        
        print(f"✅ 数据集初始化完成: {len(self)} 个样本")
        if self.gpu_preprocessing:
            print("🚀 启用GPU预处理")

    def _compute_dataset_stats(self):
        """预计算数据集统计信息"""
        if self.df is not None:
            self.real_count = len(self.df[self.df['label'] == 0])
            self.fake_count = len(self.df[self.df['label'] == 1])
        elif self.data_list is not None:
            self.real_count = sum(1 for item in self.data_list if item['label'] == 0)
            self.fake_count = sum(1 for item in self.data_list if item['label'] == 1)
        
        print(f"📊 数据分布: 真实={self.real_count}, 伪造={self.fake_count}")

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

        # 缓存检查
        cache_key = f"{video_path}_{self.max_frames}"
        if self.frame_cache is not None and cache_key in self.frame_cache:
            frames = self.frame_cache[cache_key]
            self.cache_hits += 1
        else:
            if frames is None:
                frames = extract_frames_memory_efficient(
                    video_path, 
                    max_frames=self.max_frames,
                    target_size=(224, 224)
                )
            
            # 添加到缓存
            if self.frame_cache is not None and len(self.frame_cache) < self.max_cache_size:
                self.frame_cache[cache_key] = frames
                self.cache_misses += 1

        if len(frames) == 0:
            # 创建黑色帧作为fallback
            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_frames)]

        # 确保帧数一致
        while len(frames) < self.max_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        frames = frames[:self.max_frames]

        # GPU预处理
        if self.gpu_preprocessing:
            try:
                # 转换为tensor并移动到GPU
                video_tensor = torch.stack([
                    torch.from_numpy(frame).permute(2, 0, 1) for frame in frames
                ])  # (T, C, H, W)
                
                # 移动到GPU并归一化
                video_tensor = video_tensor.to('cuda', non_blocking=True, dtype=torch.float32) / 255.0
                
                # GPU上进行数据增强
                if self.transform is None and hasattr(self, '_is_training') and self._is_training:
                    video_tensor = self._gpu_augmentation(video_tensor)
                
                # 标准化
                mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
                video_tensor = (video_tensor - mean) / std
                
                # 创建标签tensor
                label_tensor = torch.tensor(label, dtype=torch.float32)
                label_tensor = label_tensor.to('cuda', non_blocking=True)
                
                return video_tensor, label_tensor
                
            except Exception as e:
                print(f"GPU预处理失败，回退到CPU: {e}")
                # 回退到CPU处理
                pass

        # CPU处理路径
        video_tensor = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1) for frame in frames
        ]).float() / 255.0  # (T, C, H, W)

        # 应用变换
        if self.transform:
            transformed_frames = []
            for frame in video_tensor:
                frame_pil = transforms.ToPILImage()(frame)
                transformed_frame = self.transform(frame_pil)
                transformed_frames.append(transformed_frame)
            video_tensor = torch.stack(transformed_frames)
        else:
            # 默认标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            video_tensor = (video_tensor - mean) / std

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return video_tensor, label_tensor

    def _gpu_augmentation(self, video_tensor):
        """GPU上的简单数据增强"""
        # 随机水平翻转
        if torch.rand(1) > 0.5:
            video_tensor = torch.flip(video_tensor, dims=[3])
        
        # 随机亮度调整
        if torch.rand(1) > 0.5:
            brightness_factor = 0.8 + 0.4 * torch.rand(1).item()
            video_tensor = torch.clamp(video_tensor * brightness_factor, 0, 1)
        
        return video_tensor

    def get_cache_stats(self):
        """获取缓存统计信息"""
        if self.frame_cache is not None:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            return {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'cache_size': len(self.frame_cache)
            }
        return None

print("✅ 数据集类定义完成")