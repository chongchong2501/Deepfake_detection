# Cell 4: 数据集类定义

# 必要的导入
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class DeepfakeVideoDataset(Dataset):
    """深度伪造视频数据集类 - 支持预提取帧和多模态特征"""
    
    def __init__(self, csv_file, max_frames=16, gpu_preprocessing=True, 
                 extract_fourier=True, extract_compression=True, transform=None):
        """
        初始化数据集 - 专用于预提取帧的GPU预处理
        
        Args:
            csv_file: CSV文件路径（必须包含frame_path列）
            max_frames: 最大帧数
            gpu_preprocessing: 是否启用GPU预处理
            extract_fourier: 是否提取傅里叶特征
            extract_compression: 是否提取压缩特征
            transform: 数据变换（可选）
        """
        self.csv_file = csv_file
        self.max_frames = max_frames
        self.gpu_preprocessing = gpu_preprocessing
        self.extract_fourier = extract_fourier
        self.extract_compression = extract_compression
        self.transform = transform  # 添加transform属性
        
        # 加载数据
        self.df = pd.read_csv(csv_file)
        
        # 验证必须包含frame_path列
        if 'frame_path' not in self.df.columns:
            raise ValueError(f"CSV文件 {csv_file} 必须包含 'frame_path' 列。请先运行预提取流程。")
        
        print(f"✅ 预提取帧模式，共 {len(self.df)} 个样本")
        
        # GPU设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu_preprocessing else 'cpu')
        
        # 预计算的标准化参数（ImageNet标准）
        self.mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        # 预计算数据统计信息
        self._compute_dataset_stats()
        
        print(f"✅ 数据集初始化完成: {len(self)} 个样本")
        print(f"🚀 GPU预处理: {self.gpu_preprocessing} (设备: {self.device})")
        if self.extract_fourier:
            print("📊 启用频域特征提取")
        if self.extract_compression:
            print("🔍 启用压缩伪影分析")

    def _compute_dataset_stats(self):
        """预计算数据集统计信息"""
        try:
            self.real_count = len(self.df[self.df['label'] == 0])
            self.fake_count = len(self.df[self.df['label'] == 1])
        except Exception as e:
            print(f"⚠️ 计算数据统计时出错: {e}")
            self.real_count = 0
            self.fake_count = 0
        
        print(f"📊 数据分布: 真实={self.real_count}, 伪造={self.fake_count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """获取数据项 - 专用于预提取帧的GPU预处理"""
        try:
            row = self.df.iloc[idx]
            label = row['label']
            frame_path = row['frame_path']

            # 从预提取的帧文件加载
            video_tensor = self._load_preextracted_frames(frame_path)
            
            # 确保帧数一致
            video_tensor = self._ensure_frame_count(video_tensor)
            
            # GPU预处理
            if self.gpu_preprocessing and video_tensor.device != self.device:
                video_tensor = video_tensor.to(self.device, non_blocking=True)
            
            # 标准化
            video_tensor = self._normalize_frames(video_tensor)
            
            # 应用变换（如果有）
            if self.transform:
                video_tensor = self._apply_transforms(video_tensor)

            # 提取多模态特征
            additional_features = self._extract_additional_features(video_tensor)

            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            # 清理GPU内存
            if self.gpu_preprocessing:
                torch.cuda.empty_cache()
            
            # 返回数据和额外特征
            if additional_features:
                return video_tensor, label_tensor, additional_features
            else:
                return video_tensor, label_tensor
            
        except Exception as e:
            print(f"⚠️ 获取数据项 {idx} 时出错: {e}")
            # 返回默认数据
            return self._get_default_item()

    def _extract_additional_features(self, frames_tensor):
        """提取额外的多模态特征"""
        features = {}
        
        try:
            # 将tensor转换为numpy进行特征提取
            if frames_tensor.device != torch.device('cpu'):
                frames_np = frames_tensor.cpu().numpy()
            else:
                frames_np = frames_tensor.numpy()
            
            # 反标准化以获得原始像素值
            mean_np = self.mean_tensor.cpu().numpy().reshape(1, 3, 1, 1)
            std_np = self.std_tensor.cpu().numpy().reshape(1, 3, 1, 1)
            frames_np = frames_np * std_np + mean_np
            frames_np = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)
            
            if self.extract_fourier:
                # 提取频域特征（使用中间帧）
                mid_frame_idx = len(frames_np) // 2
                mid_frame = frames_np[mid_frame_idx].transpose(1, 2, 0)  # CHW -> HWC
                
                try:
                    # 检查函数是否存在
                    if 'extract_fourier_features' in globals():
                        fourier_features = extract_fourier_features(mid_frame)
                        if fourier_features:
                            features['fourier'] = fourier_features
                    else:
                        # 如果函数不存在，创建简单的频域特征替代
                        gray_frame = np.mean(mid_frame, axis=2)
                        fft = np.fft.fft2(gray_frame)
                        fft_magnitude = np.abs(fft)
                        features['fourier'] = {
                            'mean_magnitude': float(np.mean(fft_magnitude)),
                            'std_magnitude': float(np.std(fft_magnitude)),
                            'max_magnitude': float(np.max(fft_magnitude))
                        }
                except Exception as e:
                    print(f"⚠️ 频域特征提取失败: {e}")
            
            if self.extract_compression:
                # 提取压缩伪影特征
                compression_features = []
                for i in range(0, len(frames_np), 4):  # 每4帧采样一次
                    frame = frames_np[i].transpose(1, 2, 0)  # CHW -> HWC
                    try:
                        # 检查函数是否存在
                        if 'analyze_compression_artifacts' in globals():
                            comp_feat = analyze_compression_artifacts(frame)
                            if comp_feat:
                                compression_features.append(comp_feat)
                        else:
                            # 如果函数不存在，创建简单的压缩特征替代
                            gray_frame = np.mean(frame, axis=2)
                            # 简单的DCT能量计算
                            dct_energy = float(np.var(gray_frame))
                            # 简单的边缘密度计算
                            edges = np.abs(np.gradient(gray_frame.astype(float)))
                            edge_density = float(np.mean(edges[0]**2 + edges[1]**2))
                            
                            comp_feat = {
                                'dct_energy': dct_energy,
                                'edge_density': edge_density,
                                'dct_mean': dct_energy,
                                'high_freq_energy': dct_energy * 0.1
                            }
                            compression_features.append(comp_feat)
                    except Exception as e:
                        print(f"⚠️ 压缩特征提取失败: {e}")
                        continue
                
                if compression_features:
                    # 聚合压缩特征
                    features['compression'] = {
                        'dct_mean': np.mean([f.get('dct_mean', f.get('dct_energy', 0)) for f in compression_features]),
                        'dct_std': np.std([f.get('dct_mean', f.get('dct_energy', 0)) for f in compression_features]),
                        'dct_energy': np.mean([f.get('dct_energy', 0) for f in compression_features]),
                        'high_freq_energy': np.mean([f.get('high_freq_energy', f.get('dct_energy', 0) * 0.1) for f in compression_features]),
                        'edge_density': np.mean([f.get('edge_density', 0) for f in compression_features])
                    }
            
            # 计算时序一致性特征
            if len(frames_np) > 1:
                temporal_features = self._compute_temporal_consistency_tensor(frames_np)
                if temporal_features:
                    features['temporal'] = temporal_features
            
            return features if features else None
            
        except Exception as e:
            print(f"⚠️ 提取额外特征失败: {e}")
            return None

    def _compute_temporal_consistency(self, frames):
        """计算时序一致性特征（向后兼容）"""
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
    
    def _compute_temporal_consistency_tensor(self, frames_np):
        """计算时序一致性特征（tensor版本）"""
        try:
            # 计算相邻帧之间的差异
            frame_diffs = []
            for i in range(len(frames_np) - 1):
                diff = np.mean(np.abs(frames_np[i+1].astype(float) - frames_np[i].astype(float)))
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

    def _load_preextracted_frames(self, frame_path):
        """从预提取的帧文件加载数据"""
        try:
            # 直接加载tensor（数据准备阶段保存的格式）
            frames_tensor = torch.load(frame_path, map_location='cpu')
            
            # 如果加载的是字典格式，提取frames
            if isinstance(frames_tensor, dict):
                frames_tensor = frames_tensor['frames']
            
            # 确保数据类型和范围正确
            if frames_tensor.dtype != torch.float32:
                frames_tensor = frames_tensor.float()
            
            # 数据准备阶段已经将像素值标准化到[0,1]，这里需要恢复到[0,255]
            if frames_tensor.max() <= 1.0:
                frames_tensor = frames_tensor * 255.0
            
            return frames_tensor
            
        except Exception as e:
            print(f"加载预提取帧失败 {frame_path}: {e}")
            return self._create_default_frames_tensor()
    

    
    def _create_default_frames_tensor(self):
        """创建默认帧张量"""
        # 创建随机噪声帧而不是全零帧，使训练更有意义
        frames_tensor = torch.randint(0, 50, (self.max_frames, 3, 224, 224), dtype=torch.float32)
        return frames_tensor
    
    def _ensure_frame_count(self, frames_tensor):
        """确保帧数一致"""
        current_frames = frames_tensor.shape[0]
        
        if current_frames < self.max_frames:
            # 重复最后一帧
            last_frame = frames_tensor[-1:]
            repeat_count = self.max_frames - current_frames
            repeated_frames = last_frame.repeat(repeat_count, 1, 1, 1)
            frames_tensor = torch.cat([frames_tensor, repeated_frames], dim=0)
        elif current_frames > self.max_frames:
            # 截取前max_frames帧
            frames_tensor = frames_tensor[:self.max_frames]
        
        return frames_tensor
    
    def _normalize_frames(self, frames_tensor):
        """标准化帧数据"""
        # 确保像素值在[0, 1]范围内
        if frames_tensor.max() > 1.0:
            frames_tensor = frames_tensor / 255.0
        
        # 移动标准化参数到正确设备
        if self.mean_tensor.device != frames_tensor.device:
            self.mean_tensor = self.mean_tensor.to(frames_tensor.device)
            self.std_tensor = self.std_tensor.to(frames_tensor.device)
        
        # ImageNet标准化
        frames_tensor = (frames_tensor - self.mean_tensor) / self.std_tensor
        
        # 限制数值范围防止梯度爆炸
        frames_tensor = torch.clamp(frames_tensor, -10, 10)
        
        return frames_tensor
    
    def _apply_transforms(self, frames_tensor):
        """应用数据变换"""
        try:
            # 将tensor转换回PIL格式进行变换
            transformed_frames = []
            
            # 反标准化以获得原始像素值
            denorm_tensor = frames_tensor * self.std_tensor + self.mean_tensor
            denorm_tensor = torch.clamp(denorm_tensor * 255.0, 0, 255)
            
            for i in range(frames_tensor.shape[0]):
                frame = denorm_tensor[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                frame_pil = Image.fromarray(frame)
                transformed_frame = self.transform(frame_pil)
                
                # 检查变换后是否有NaN或无穷值
                if torch.isnan(transformed_frame).any() or torch.isinf(transformed_frame).any():
                    print(f"⚠️ 检测到NaN/Inf值，跳过变换")
                    return frames_tensor
                
                transformed_frames.append(transformed_frame)
            
            return torch.stack(transformed_frames)
            
        except Exception as e:
            print(f"⚠️ 数据变换失败，使用原始数据: {e}")
            return frames_tensor
    


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

    def _create_default_frames(self):
        """创建默认帧数据（numpy格式）"""
        # 创建随机噪声帧而不是全零帧，使训练更有意义
        frames = []
        for _ in range(self.max_frames):
            # 创建224x224x3的随机帧，值在[0, 50]范围内（低噪声）
            frame = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
            frames.append(frame)
        return frames



    def enable_ensemble_mode(self):
        """启用集成模式，提取所有可用特征"""
        self.extract_fourier = True
        self.extract_compression = True
        print("🎯 启用集成模式：所有特征提取已激活")

print("✅ 数据集类定义完成")