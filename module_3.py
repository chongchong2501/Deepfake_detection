#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 第3段：优化的数据集类定义
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

# =============================================================================
# 第3段：优化的数据集类定义
# =============================================================================

import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from collections import Counter

class AdvancedDeepfakeDataset(Dataset):
    """
    高级深度伪造检测数据集（无imgaug依赖版本）
    """
    def __init__(self, csv_file, transform=None, max_frames=32, 
                 augment_prob=0.5, temporal_augment=True, 
                 mixup_alpha=0.2, cutmix_alpha=1.0, mode='train'):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.max_frames = max_frames
        self.augment_prob = augment_prob
        self.temporal_augment = temporal_augment
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mode = mode
        
        # 创建类别权重（用于处理不平衡数据）
        self.class_weights = self._calculate_class_weights()
        
        # 初始化数据增强
        self._init_augmentations()
        
        print(f"数据集初始化完成: {len(self.data)} 个样本 ({mode} 模式)")
        print(f"真实视频: {len(self.data[self.data['label']==0])} 个")
        print(f"伪造视频: {len(self.data[self.data['label']==1])} 个")
    
    def _calculate_class_weights(self):
        """计算类别权重"""
        class_counts = self.data['label'].value_counts().sort_index()
        total_samples = len(self.data)
        weights = total_samples / (len(class_counts) * class_counts.values)
        return torch.FloatTensor(weights)
    
    def _init_augmentations(self):
        """初始化数据增强（使用torchvision替代imgaug）"""
        # 空间增强（使用torchvision）
        self.spatial_transforms = [
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.RandomAutocontrast(p=0.2),
            T.RandomEqualize(p=0.2),
        ]
        
        # 时序增强
        self.temporal_augs = {
            'frame_drop': 0.1,      # 随机丢弃帧
            'frame_repeat': 0.1,    # 随机重复帧
            'temporal_shift': 0.2,  # 时序偏移
            'reverse': 0.05,        # 时序反转
        }
    
    def __len__(self):
        return len(self.data)
    
    def _load_and_preprocess_frames(self, frame_path):
        """加载和预处理帧数据"""
        try:
            frames = np.load(frame_path)
            
            # 确保帧数量
            if len(frames) < self.max_frames:
                # 智能填充：使用插值而不是简单重复
                indices = np.linspace(0, len(frames)-1, self.max_frames)
                new_frames = []
                for i in indices:
                    if i == int(i):
                        new_frames.append(frames[int(i)])
                    else:
                        # 线性插值
                        i1, i2 = int(i), min(int(i)+1, len(frames)-1)
                        alpha = i - i1
                        frame = (1-alpha) * frames[i1] + alpha * frames[i2]
                        new_frames.append(frame.astype(np.uint8))
                frames = np.array(new_frames)
            elif len(frames) > self.max_frames:
                # 智能采样：保留关键帧
                indices = self._select_key_frames(frames, self.max_frames)
                frames = frames[indices]
            
            return frames
        except Exception as e:
            print(f"加载帧数据失败 {frame_path}: {e}")
            # 返回随机帧作为fallback
            return np.random.randint(0, 255, (self.max_frames, 224, 224, 3), dtype=np.uint8)
    
    def _select_key_frames(self, frames, target_count):
        """选择关键帧"""
        if len(frames) <= target_count:
            return np.arange(len(frames))
        
        # 计算帧间差异
        frame_diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            frame_diffs.append(diff)
        
        # 选择变化最大的帧
        key_indices = [0]  # 总是包含第一帧
        
        # 基于差异选择帧
        remaining_count = target_count - 2  # 减去首尾帧
        if remaining_count > 0:
            diff_indices = np.argsort(frame_diffs)[-remaining_count:]
            key_indices.extend(sorted(diff_indices + 1))  # +1因为diff_indices是相对于frames[1:]的
        
        key_indices.append(len(frames) - 1)  # 总是包含最后一帧
        
        return sorted(list(set(key_indices)))[:target_count]
    
    def _apply_temporal_augmentation(self, frames):
        """应用时序增强"""
        if not self.temporal_augment or self.mode != 'train':
            return frames
        
        frames = frames.copy()
        
        # 随机丢弃帧
        if random.random() < self.temporal_augs['frame_drop']:
            drop_count = random.randint(1, min(3, len(frames)//4))
            drop_indices = random.sample(range(len(frames)), drop_count)
            for idx in sorted(drop_indices, reverse=True):
                if len(frames) > self.max_frames // 2:  # 确保不会丢弃太多帧
                    frames = np.delete(frames, idx, axis=0)
        
        # 随机重复帧
        if random.random() < self.temporal_augs['frame_repeat']:
            repeat_idx = random.randint(0, len(frames)-1)
            frames = np.insert(frames, repeat_idx, frames[repeat_idx], axis=0)
        
        # 时序偏移
        if random.random() < self.temporal_augs['temporal_shift']:
            shift = random.randint(-2, 2)
            if shift != 0:
                frames = np.roll(frames, shift, axis=0)
        
        # 时序反转
        if random.random() < self.temporal_augs['reverse']:
            frames = frames[::-1]
        
        # 确保帧数量
        if len(frames) != self.max_frames:
            if len(frames) < self.max_frames:
                # 重复最后几帧
                repeat_count = self.max_frames - len(frames)
                last_frames = frames[-repeat_count:]
                frames = np.concatenate([frames, last_frames], axis=0)
            else:
                # 截断
                frames = frames[:self.max_frames]
        
        return frames
    
    def _apply_spatial_augmentation(self, frames):
        """应用空间增强（使用torchvision替代imgaug）"""
        if self.mode != 'train' or random.random() > self.augment_prob:
            return frames
        
        # 对每一帧应用增强
        augmented_frames = []
        for frame in frames:
            if random.random() < 0.7:  # 70%的概率对单帧进行增强
                # 转换为PIL图像
                frame_pil = T.ToPILImage()(torch.tensor(frame).permute(2, 0, 1))
                
                # 随机选择一个增强
                if self.spatial_transforms:
                    aug = random.choice(self.spatial_transforms)
                    frame_pil = aug(frame_pil)
                
                # 转换回numpy
                frame = np.array(frame_pil)
            
            augmented_frames.append(frame)
        
        return np.array(augmented_frames)
    
    def _apply_mixup(self, frames1, label1, frames2, label2):
        """应用MixUp增强"""
        if self.mixup_alpha <= 0:
            return frames1, label1
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed_frames = lam * frames1 + (1 - lam) * frames2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_frames, mixed_label
    
    def _apply_cutmix(self, frames, label):
        """应用CutMix增强"""
        if self.cutmix_alpha <= 0 or self.mode != 'train':
            return frames, label
        
        # 随机选择另一个样本
        mix_idx = random.randint(0, len(self.data) - 1)
        mix_row = self.data.iloc[mix_idx]
        mix_frames = self._load_and_preprocess_frames(mix_row['frame_path'])
        mix_label = mix_row['label']
        
        # 应用CutMix
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        H, W = frames.shape[1], frames.shape[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        frames[:, bby1:bby2, bbx1:bbx2, :] = mix_frames[:, bby1:bby2, bbx1:bbx2, :]
        
        # 调整标签
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_label = lam * label + (1 - lam) * mix_label
        
        return frames, mixed_label
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 加载帧数据
        frames = self._load_and_preprocess_frames(row['frame_path'])
        label = float(row['label'])
        
        # 应用时序增强
        frames = self._apply_temporal_augmentation(frames)
        
        # 应用空间增强
        frames = self._apply_spatial_augmentation(frames)
        
        # 应用CutMix（训练时）
        if self.mode == 'train' and random.random() < 0.1:
            frames, label = self._apply_cutmix(frames, label)
        
        # 转换为tensor
        if self.transform:
            transformed_frames = []
            for frame in frames:
                # 确保frame是uint8类型
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            frames = torch.stack(transformed_frames)
        else:
            frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        
        # 添加额外的元数据
        metadata = {
            'video_name': row.get('video_name', ''),
            'method': row.get('method', ''),
            'avg_quality': row.get('avg_quality', 0.0)
        }
        
        return frames, torch.tensor(label, dtype=torch.float32), metadata

# 创建加权采样器
def create_weighted_sampler(dataset):
    """创建加权随机采样器以处理类别不平衡"""
    labels = [dataset.data.iloc[i]['label'] for i in range(len(dataset))]
    class_counts = Counter(labels)
    
    # 计算每个样本的权重
    weights = []
    for label in labels:
        weight = 1.0 / class_counts[label]
        weights.append(weight)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# 优化的数据变换
def get_optimized_transforms(mode='train', image_size=224):
    """获取优化的数据变换"""
    if mode == 'train':
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2)
            ], p=0.2),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    return lambda x: transform(image=x)['image']

# 创建数据加载器
def create_optimized_dataloaders(train_csv, val_csv, test_csv=None, 
                               batch_size=16, num_workers=4, 
                               max_frames=32, image_size=224):
    """创建优化的数据加载器"""
    
    # 获取变换
    train_transform = get_optimized_transforms('train', image_size)
    val_transform = get_optimized_transforms('val', image_size)
    
    # 创建数据集
    train_dataset = AdvancedDeepfakeDataset(
        train_csv, transform=train_transform, max_frames=max_frames,
        augment_prob=0.6, temporal_augment=True, mode='train'
    )
    
    val_dataset = AdvancedDeepfakeDataset(
        val_csv, transform=val_transform, max_frames=max_frames,
        augment_prob=0.0, temporal_augment=False, mode='val'
    )
    
    # 创建采样器
    train_sampler = create_weighted_sampler(train_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    loaders = {'train': train_loader, 'val': val_loader}
    
    # 测试集（如果提供）
    if test_csv and os.path.exists(test_csv):
        test_dataset = AdvancedDeepfakeDataset(
            test_csv, transform=val_transform, max_frames=max_frames,
            augment_prob=0.0, temporal_augment=False, mode='test'
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        loaders['test'] = test_loader
    
    return loaders

print("✅ 数据集定义完成")