# Cell 4: 数据集类定义

class DeepfakeVideoDataset(Dataset):
    """深度伪造视频数据集类 - GPU优化版本"""
    
    def __init__(self, csv_file=None, data_list=None, transform=None, max_frames=32, gpu_preprocessing=True):
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
        
        # GPU预处理的标准化参数
        if self.gpu_preprocessing:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
            self.std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    
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
        else:
            # 从CSV文件获取路径并重新提取帧
            row = self.df.iloc[idx]
            video_path = row['video_path']
            label = row['label']
            frames = extract_frames_memory_efficient(video_path, self.max_frames)
        
        # 确保有足够的帧
        if len(frames) == 0:
            # 创建黑色帧作为fallback
            frames = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(self.max_frames)]  # 降低分辨率
        
        while len(frames) < self.max_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((128, 128, 3), dtype=np.uint8))
        
        frames = frames[:self.max_frames]
        
        # GPU优化的预处理
        if self.gpu_preprocessing and not self.transform:
            # 快速转换为tensor并移到GPU
            frames_array = np.stack(frames)  # (T, H, W, C)
            video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
            video_tensor = video_tensor.cuda(non_blocking=True) / 255.0
            
            # GPU上进行标准化
            video_tensor = (video_tensor - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
        else:
            # 传统CPU预处理
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            else:
                # 默认变换
                frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
            
            # 堆叠帧 (T, C, H, W)
            video_tensor = torch.stack(frames)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return video_tensor, label_tensor

print("✅ 数据集类定义完成")