# Cell 10: 数据加载器创建
def create_data_loaders(train_df, val_df, test_df, batch_size=16, num_workers=2, 
                       balance_classes=True, oversample_minority=True):
    """
    创建数据加载器 - 增强类别平衡
    
    Args:
        balance_classes: 是否平衡类别
        oversample_minority: 是否对少数类进行过采样
    """
    print("📊 创建数据加载器...")
    
    # 获取数据变换 - 修复版本
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')
    
    # 创建数据集 - 启用多模态特征提取
    train_dataset = DeepfakeVideoDataset(
        data_list=train_df.to_dict('records'), 
        transform=train_transform,
        extract_fourier=True,  # 启用频域特征
        extract_compression=True  # 启用压缩特征
    )
    val_dataset = DeepfakeVideoDataset(
        data_list=val_df.to_dict('records'), 
        transform=val_transform,
        extract_fourier=True,  # 启用频域特征
        extract_compression=True  # 启用压缩特征
    )
    test_dataset = DeepfakeVideoDataset(
        data_list=test_df.to_dict('records'), 
        transform=val_transform,
        extract_fourier=True,  # 启用频域特征
        extract_compression=True  # 启用压缩特征
    )
    
    # 分析类别分布
    train_labels = train_df['label'].values
    real_count = np.sum(train_labels == 0)
    fake_count = np.sum(train_labels == 1)
    total_count = len(train_labels)
    
    print(f"📈 训练数据分布:")
    print(f"   - 真实视频: {real_count} ({real_count/total_count*100:.1f}%)")
    print(f"   - 伪造视频: {fake_count} ({fake_count/total_count*100:.1f}%)")
    print(f"   - 不平衡比例: {max(real_count, fake_count)/min(real_count, fake_count):.2f}:1")
    
    # 创建采样器
    train_sampler = None
    if balance_classes and abs(real_count - fake_count) > total_count * 0.1:  # 如果不平衡超过10%
        print("⚖️ 检测到类别不平衡，应用平衡采样...")
        
        if oversample_minority:
            # 过采样少数类
            from torch.utils.data import WeightedRandomSampler
            class_counts = [real_count, fake_count]
            class_weights = [1.0 / count for count in class_counts]
            sample_weights = [class_weights[label] for label in train_labels]
            
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            print(f"   ✅ 使用加权随机采样器")
        else:
            # 下采样多数类
            from torch.utils.data import Subset
            
            real_indices = np.where(train_labels == 0)[0]
            fake_indices = np.where(train_labels == 1)[0]
            
            min_count = min(real_count, fake_count)
            balanced_real_indices = np.random.choice(real_indices, min_count, replace=False)
            balanced_fake_indices = np.random.choice(fake_indices, min_count, replace=False)
            
            balanced_indices = np.concatenate([balanced_real_indices, balanced_fake_indices])
            np.random.shuffle(balanced_indices)
            
            train_dataset = Subset(train_dataset, balanced_indices)
            print(f"   ✅ 下采样到平衡数据集: {len(balanced_indices)} 样本")
    
    # 创建数据加载器 - 修复多进程序列化问题
    # 在Jupyter/Kaggle环境中，使用num_workers=0避免序列化问题
    safe_num_workers = 0  # 强制使用单进程模式
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # 如果有采样器就不shuffle
        num_workers=safe_num_workers,
        pin_memory=True,
        drop_last=True,  # 确保批次大小一致
        persistent_workers=False  # 单进程模式下不需要
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_num_workers,
        pin_memory=True,
        persistent_workers=False  # 单进程模式下不需要
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_num_workers,
        pin_memory=True,
        persistent_workers=False  # 单进程模式下不需要
    )
    
    print(f"✅ 数据加载器创建完成:")
    print(f"   - 训练批次: {len(train_loader)}")
    print(f"   - 验证批次: {len(val_loader)}")
    print(f"   - 测试批次: {len(test_loader)}")
    print(f"   - 工作进程: {safe_num_workers} (单进程模式，避免序列化问题)")
    
    return train_loader, val_loader, test_loader

# 检查数据文件是否存在，如果存在则加载数据
if os.path.exists('./data/train.csv') and os.path.exists('./data/val.csv') and os.path.exists('./data/test.csv'):
    print("📊 加载现有数据集...")
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/val.csv')
    test_df = pd.read_csv('./data/test.csv')
    
    # 创建数据加载器实例 - 使用全局配置参数，但强制单进程模式
    print("🔄 正在创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df=train_df, 
        val_df=val_df, 
        test_df=test_df,
        batch_size=2,  # 
        num_workers=0,  # 强制使用单进程模式避免序列化问题
        balance_classes=True,
        oversample_minority=True
    )
    
    print("✅ 数据加载器创建完成，可以开始训练了！")
else:
    print("⚠️ 数据文件不存在，请先运行数据准备步骤（cell_09）")
    train_loader = val_loader = test_loader = None