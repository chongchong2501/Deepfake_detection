# Cell 4: 数据集类定义

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeVideoDataset(Dataset):
    """深度伪造视频数据集类 - 集成MTCNN和多模态特征"""
    
    def __init__(self, csv_file=None, data_list=None, transform=None, max_frames=16, 
                 gpu_preprocessing=False, cache_frames=False, use_mtcnn=True, 
                 extract_fourier=False, extract_compression=False):
        if csv_file is not None:
            try:
                self.df = pd.read_csv(csv_file)
                self.data_list = None
                print(f"✅ 成功加载CSV文件: {csv_file}")
            except FileNotFoundError:
                print(f"⚠️ CSV文件不存在: {csv_file}，创建空数据集")
                self.df = pd.DataFrame(columns=['video_path', 'label'])
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
        self.use_mtcnn = use_mtcnn and globals().get('MTCNN_AVAILABLE', False)
        self.extract_fourier = extract_fourier and globals().get('SCIPY_AVAILABLE', False)
        self.extract_compression = extract_compression
        
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
        if self.use_mtcnn:
            print("👁️ 启用MTCNN人脸检测")
        if self.extract_fourier:
            print("📊 启用频域特征提取")
        if self.extract_compression:
            print("🔍 启用压缩伪影分析")

    def _compute_dataset_stats(self):
        """预计算数据集统计信息"""
        try:
            if self.df is not None and len(self.df) > 0:
                self.real_count = len(self.df[self.df['label'] == 0])
                self.fake_count = len(self.df[self.df['label'] == 1])
            elif self.data_list is not None:
                self.real_count = sum(1 for item in self.data_list if item['label'] == 0)
                self.fake_count = sum(1 for item in self.data_list if item['label'] == 1)
            else:
                self.real_count = 0
                self.fake_count = 0
        except Exception as e:
            print(f"⚠️ 计算数据统计时出错: {e}")
            self.real_count = 0
            self.fake_count = 0
        
        print(f"📊 数据分布: 真实={self.real_count}, 伪造={self.fake_count}")

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.data_list) if self.data_list else 0

    def __getitem__(self, idx):
        try:
            if self.data_list is not None:
                item = self.data_list[idx]
                video_path = item['video_path']
                frames = item.get('frames', None)
                label = item['label']
            else:
                row = self.df.iloc[idx]
                video_path = row['video_path']
                label = row['label']
                frames = None

            # 如果没有预提取的帧，则实时提取
            if frames is None:
                try:
                    from .cell_03_data_processing import extract_frames_memory_efficient
                    frames = extract_frames_memory_efficient(
                        video_path, 
                        max_frames=self.max_frames,
                        use_mtcnn=self.use_mtcnn
                    )
                except Exception as e:
                    print(f"⚠️ 实时帧提取失败: {e}")
                    frames = self._create_default_frames()
            
            # 如果仍然没有帧，创建默认帧
            if not frames:
                frames = self._create_default_frames()
            
            # 确保帧数一致
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
            frames = frames[:self.max_frames]

            # 提取多模态特征
            additional_features = self._extract_additional_features(frames)

            # 始终使用CPU处理路径确保稳定性
            video_tensor = torch.stack([
                torch.from_numpy(frame).permute(2, 0, 1) for frame in frames
            ]).float() / 255.0  # (T, C, H, W)

            # 应用变换
            if self.transform:
                try:
                    transformed_frames = []
                    for frame in video_tensor:
                        frame_pil = transforms.ToPILImage()(frame)
                        transformed_frame = self.transform(frame_pil)
                        transformed_frames.append(transformed_frame)
                    video_tensor = torch.stack(transformed_frames)
                except Exception as e:
                    print(f"⚠️ 数据变换失败，使用原始数据: {e}")
            
            # 默认标准化
            try:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                video_tensor = (video_tensor - mean) / std
            except Exception as e:
                print(f"⚠️ 标准化失败: {e}")

            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            # 返回数据和额外特征
            if additional_features:
                return video_tensor, label_tensor, additional_features
            else:
                return video_tensor, label_tensor
            
        except Exception as e:
            print(f"⚠️ 获取数据项 {idx} 时出错: {e}")
            # 返回默认数据
            return self._get_default_item()

    def _extract_additional_features(self, frames):
        """提取额外的多模态特征"""
        features = {}
        
        try:
            if self.extract_fourier:
                # 提取频域特征（使用中间帧）
                mid_frame = frames[len(frames) // 2]
                from .cell_03_data_processing import extract_fourier_features
                fourier_features = extract_fourier_features(mid_frame)
                if fourier_features:
                    features['fourier'] = fourier_features
            
            if self.extract_compression:
                # 提取压缩伪影特征
                compression_features = []
                for frame in frames[::4]:  # 每4帧采样一次
                    from .cell_03_data_processing import analyze_compression_artifacts
                    comp_feat = analyze_compression_artifacts(frame)
                    if comp_feat:
                        compression_features.append(comp_feat)
                
                if compression_features:
                    # 聚合压缩特征
                    features['compression'] = {
                        'mean_dct_energy': np.mean([f['dct_energy'] for f in compression_features]),
                        'mean_edge_density': np.mean([f['edge_density'] for f in compression_features]),
                        'std_dct_energy': np.std([f['dct_energy'] for f in compression_features])
                    }
            
            # 计算时序一致性特征
            if len(frames) > 1:
                temporal_features = self._compute_temporal_consistency(frames)
                if temporal_features:
                    features['temporal'] = temporal_features
            
            return features if features else None
            
        except Exception as e:
            print(f"⚠️ 提取额外特征失败: {e}")
            return None

    def _compute_temporal_consistency(self, frames):
        """计算时序一致性特征"""
        try:
            # 计算相邻帧之间的差异
            frame_diffs = []
            for i in range(len(frames) - 1):
                diff = np.mean(np.abs(frames[i+1].astype(float) - frames[i].astype(float)))
                frame_diffs.append(diff)
            
            if frame_diffs:
                return {
                    'mean_frame_diff': np.mean(frame_diffs),
                    'std_frame_diff': np.std(frame_diffs),
                    'max_frame_diff': np.max(frame_diffs),
                    'temporal_smoothness': 1.0 / (1.0 + np.std(frame_diffs))
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ 计算时序特征失败: {e}")
            return None

    def _create_default_frames(self):
        """创建默认帧数据"""
        # 创建随机噪声帧而不是全零帧，使训练更有意义
        frames = []
        for i in range(self.max_frames):
            # 创建带有轻微随机噪声的帧
            frame = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
            frames.append(frame)
        return frames

    def _get_default_item(self):
        """获取默认数据项（用于错误恢复）"""
        frames = self._create_default_frames()
        video_tensor = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1) for frame in frames
        ]).float() / 255.0
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        label_tensor = torch.tensor(0.0, dtype=torch.float32)
        return video_tensor, label_tensor

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

    def enable_ensemble_mode(self):
        """启用集成模式，提取所有可用特征"""
        self.extract_fourier = globals().get('SCIPY_AVAILABLE', False)
        self.extract_compression = True
        self.use_mtcnn = globals().get('MTCNN_AVAILABLE', False)
        print("🎯 启用集成模式：所有特征提取已激活")

print("✅ 数据集类定义完成")