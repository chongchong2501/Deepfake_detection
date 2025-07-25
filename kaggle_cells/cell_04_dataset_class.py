# Cell 4: 数据集类定义

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
                    for frame in frames:  # 直接使用原始numpy帧
                        # 确保frame是numpy数组且为uint8类型
                        if isinstance(frame, np.ndarray):
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                            # 转换为PIL Image
                            frame_pil = Image.fromarray(frame)
                            transformed_frame = self.transform(frame_pil)
                            
                            # 检查变换后是否有NaN或无穷值
                            if torch.isnan(transformed_frame).any() or torch.isinf(transformed_frame).any():
                                print(f"⚠️ 检测到NaN/Inf值，使用默认帧")
                                default_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                                frame_pil = Image.fromarray(default_frame)
                                transformed_frame = self.transform(frame_pil)
                            
                            transformed_frames.append(transformed_frame)
                        else:
                            # 如果不是numpy数组，创建默认帧
                            default_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                            frame_pil = Image.fromarray(default_frame)
                            transformed_frame = self.transform(frame_pil)
                            transformed_frames.append(transformed_frame)
                    video_tensor = torch.stack(transformed_frames)
                    
                    # 最终检查整个视频张量
                    if torch.isnan(video_tensor).any() or torch.isinf(video_tensor).any():
                        print(f"⚠️ 视频张量包含NaN/Inf，使用安全的默认处理")
                        # 回退到原始处理方式
                        video_tensor = torch.stack([
                            torch.from_numpy(frame).permute(2, 0, 1) for frame in frames
                        ]).float() / 255.0
                        # 手动应用标准化，使用更安全的数值
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                        video_tensor = torch.clamp((video_tensor - mean) / std, -10, 10)  # 限制数值范围
                        
                except Exception as e:
                    print(f"⚠️ 数据变换失败，使用原始数据: {e}")
                    # 回退到原始处理方式，但不再应用标准化（因为transform中已包含）
                    video_tensor = torch.stack([
                        torch.from_numpy(frame).permute(2, 0, 1) for frame in frames
                    ]).float() / 255.0
                    # 手动应用标准化，使用更安全的数值
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    video_tensor = torch.clamp((video_tensor - mean) / std, -10, 10)  # 限制数值范围
            else:
                # 如果没有变换，应用默认标准化，使用更安全的数值处理
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                video_tensor = torch.clamp((video_tensor - mean) / std, -10, 10)  # 限制数值范围

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
                try:
                    # 检查函数是否存在
                    if 'extract_fourier_features' in globals():
                        fourier_features = extract_fourier_features(mid_frame)
                        if fourier_features:
                            features['fourier'] = fourier_features
                    else:
                        # 如果函数不存在，创建简单的频域特征替代
                        # 使用FFT计算简单的频域统计
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
                    # 跳过频域特征
            
            if self.extract_compression:
                # 提取压缩伪影特征
                compression_features = []
                for frame in frames[::4]:  # 每4帧采样一次
                    try:
                        # 检查函数是否存在
                        if 'analyze_compression_artifacts' in globals():
                            comp_feat = analyze_compression_artifacts(frame)
                            if comp_feat:
                                compression_features.append(comp_feat)
                        else:
                            # 如果函数不存在，创建简单的压缩特征替代
                            # 使用DCT和边缘检测的简单实现
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
                    # 聚合压缩特征 - 使用与模型期望匹配的键名
                    features['compression'] = {
                        'dct_mean': np.mean([f.get('dct_mean', f.get('dct_energy', 0)) for f in compression_features]),
                        'dct_std': np.std([f.get('dct_mean', f.get('dct_energy', 0)) for f in compression_features]),
                        'dct_energy': np.mean([f.get('dct_energy', 0) for f in compression_features]),
                        'high_freq_energy': np.mean([f.get('high_freq_energy', f.get('dct_energy', 0) * 0.1) for f in compression_features]),
                        'edge_density': np.mean([f.get('edge_density', 0) for f in compression_features])
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