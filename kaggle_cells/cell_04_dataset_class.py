# Cell 4: 数据集类定义

class DeepfakeVideoDataset(Dataset):
    """深度伪造视频数据集类 - 全GPU加速版本"""
    
    def __init__(self, csv_file=None, data_list=None, transform=None, max_frames=32, 
                 gpu_preprocessing=True, cache_frames=True, full_gpu_pipeline=True):
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
        self.full_gpu_pipeline = full_gpu_pipeline and torch.cuda.is_available()
        
        # 多级缓存系统
        self.frame_cache = {} if cache_frames else None  # CPU缓存
        self.gpu_cache = {} if (cache_frames and self.full_gpu_pipeline) else None  # GPU缓存
        self.cache_hits = 0
        self.cache_misses = 0
        
        # GPU预处理的标准化参数
        if self.gpu_preprocessing:
            self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=torch.float16)
            self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda', dtype=torch.float16)
            
        # GPU内存管理
        if self.full_gpu_pipeline:
            self.max_gpu_cache_size = 100  # 最大GPU缓存视频数
            self.gpu_cache_keys = []  # LRU缓存键列表
            
        print(f"🚀 数据集初始化: GPU流水线={'启用' if self.full_gpu_pipeline else '禁用'}, "
              f"GPU预处理={'启用' if self.gpu_preprocessing else '禁用'}, "
              f"缓存={'启用' if self.cache_frames else '禁用'}")
    
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
            video_path = None
        else:
            # 从CSV文件获取路径
            row = self.df.iloc[idx]
            video_path = row['video_path']
            label = row['label']
            frames = None
        
        # 全GPU流水线处理
        if self.full_gpu_pipeline and video_path is not None:
            return self._process_video_gpu_pipeline(video_path, label)
        
        # 传统处理流程（兼容性）
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
        
        # 确保有足够的帧
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_frames)]
        
        while len(frames) < self.max_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        frames = frames[:self.max_frames]
        
        # 高效GPU预处理流水线
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            video_tensor = torch.stack(frames)
        elif self.gpu_preprocessing:
            # 批量GPU预处理：减少CPU-GPU传输次数
            frames_array = np.stack(frames)  # (T, H, W, C)
            video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
            
            # 一次性移动到GPU并进行所有预处理
            video_tensor = video_tensor.to('cuda', non_blocking=True, dtype=torch.float16) / 255.0
            
            # 使用预计算的标准化参数进行高效标准化
            video_tensor = (video_tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
        else:
            # CPU预处理（保持原有逻辑）
            frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
            video_tensor = torch.stack(frames)
        
        # 标签处理
        if self.gpu_preprocessing and not self.transform:
            label_tensor = torch.tensor(label, dtype=torch.float32, device='cuda')
        else:
            label_tensor = torch.tensor(label, dtype=torch.float32)
            if self.gpu_preprocessing:
                video_tensor = video_tensor.to('cuda', non_blocking=True)
                label_tensor = label_tensor.to('cuda', non_blocking=True)
        
        return video_tensor, label_tensor
    
    def _process_video_gpu_pipeline(self, video_path, label):
        """完全GPU端到端的视频处理流水线"""
        # 检查GPU缓存
        if self.gpu_cache is not None and video_path in self.gpu_cache:
            self.cache_hits += 1
            self._update_gpu_cache_lru(video_path)
            video_tensor = self.gpu_cache[video_path]
            label_tensor = torch.tensor(label, dtype=torch.float32, device='cuda')
            return video_tensor, label_tensor
        
        self.cache_misses += 1
        
        try:
            # 使用torchvision直接在GPU上读取和处理视频
            # import语句已在cell_01中定义
            
            # 读取视频
            video_tensor, _, _ = read_video(video_path, pts_unit='sec')
            
            if video_tensor.size(0) == 0:
                # 创建黑色帧
                video_tensor = torch.zeros(self.max_frames, 224, 224, 3, dtype=torch.float16, device='cuda')
            else:
                # 移动到GPU
                video_tensor = video_tensor.to('cuda', non_blocking=True, dtype=torch.float16)
                
                # 智能帧采样
                total_frames = video_tensor.size(0)
                if total_frames <= self.max_frames:
                    frame_indices = torch.arange(0, total_frames, device='cuda')
                else:
                    step = total_frames / self.max_frames
                    frame_indices = torch.arange(0, total_frames, step, device='cuda').long()[:self.max_frames]
                
                video_tensor = video_tensor[frame_indices]  # (T, H, W, C)
                
                # 确保帧数足够
                current_frames = video_tensor.size(0)
                if current_frames < self.max_frames:
                    if current_frames > 0:
                        last_frame = video_tensor[-1:].repeat(self.max_frames - current_frames, 1, 1, 1)
                        video_tensor = torch.cat([video_tensor, last_frame], dim=0)
                    else:
                        video_tensor = torch.zeros(self.max_frames, 224, 224, 3, dtype=torch.float16, device='cuda')
                
                # 调整尺寸到224x224
                video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
                if video_tensor.size(-1) != 224 or video_tensor.size(-2) != 224:
                    video_tensor = F.interpolate(video_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            
            # GPU上进行标准化
            video_tensor = video_tensor / 255.0
            video_tensor = (video_tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
            
            # 限制到最大帧数
            video_tensor = video_tensor[:self.max_frames]
            
            # 添加到GPU缓存
            if self.gpu_cache is not None:
                self._add_to_gpu_cache(video_path, video_tensor)
            
            label_tensor = torch.tensor(label, dtype=torch.float32, device='cuda')
            return video_tensor, label_tensor
            
        except Exception as e:
            print(f"GPU流水线处理失败，回退到传统方法: {e}")
            # 回退到传统CPU处理
            frames = extract_frames_gpu_accelerated(video_path, self.max_frames, target_size=(224, 224))
            if len(frames) == 0:
                frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.max_frames)]
            
            while len(frames) < self.max_frames:
                frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8))
            
            frames = frames[:self.max_frames]
            frames_array = np.stack(frames)
            video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()
            video_tensor = video_tensor.to('cuda', non_blocking=True, dtype=torch.float16) / 255.0
            video_tensor = (video_tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
            
            label_tensor = torch.tensor(label, dtype=torch.float32, device='cuda')
            return video_tensor, label_tensor
    
    def _add_to_gpu_cache(self, video_path, video_tensor):
        """添加视频到GPU缓存，使用LRU策略"""
        if len(self.gpu_cache) >= self.max_gpu_cache_size:
            # 移除最旧的缓存项
            oldest_key = self.gpu_cache_keys.pop(0)
            del self.gpu_cache[oldest_key]
        
        self.gpu_cache[video_path] = video_tensor.clone()
        self.gpu_cache_keys.append(video_path)
    
    def _update_gpu_cache_lru(self, video_path):
        """更新GPU缓存的LRU顺序"""
        if video_path in self.gpu_cache_keys:
            self.gpu_cache_keys.remove(video_path)
            self.gpu_cache_keys.append(video_path)
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'gpu_cache_size': len(self.gpu_cache) if self.gpu_cache else 0,
            'cpu_cache_size': len(self.frame_cache) if self.frame_cache else 0
        }

print("✅ 数据集类定义完成")