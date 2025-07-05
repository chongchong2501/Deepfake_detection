# Cell 4: 数据集类定义

class DeepfakeVideoDataset(Dataset):
    """深度伪造视频数据集类"""
    
    def __init__(self, csv_file=None, data_list=None, transform=None, max_frames=32):
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
            frames = [np.zeros((160, 160, 3), dtype=np.uint8) for _ in range(self.max_frames)]
        
        while len(frames) < self.max_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((160, 160, 3), dtype=np.uint8))
        
        frames = frames[:self.max_frames]
        
        # 应用变换
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