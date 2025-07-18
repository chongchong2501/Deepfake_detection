# Cell 10: 创建数据加载器 - Kaggle T4 优化版本

print("📊 创建数据加载器...")

# 简化数据变换 - 使用GPU预处理替代CPU变换
train_transform = None
val_transform = None

print(f"🔧 创建数据集（Kaggle T4优化配置）...")
print(f"📊 数据类型: FP32 (兼容性优先)")

# 创建数据集
train_dataset = DeepfakeVideoDataset(
    csv_file='./data/train.csv',
    transform=train_transform,
    max_frames=16,
    gpu_preprocessing=True,
    cache_frames=False  # 避免内存压力
)

val_dataset = DeepfakeVideoDataset(
    csv_file='./data/val.csv',
    transform=val_transform,
    max_frames=16,
    gpu_preprocessing=True,
    cache_frames=False
)

test_dataset = DeepfakeVideoDataset(
    csv_file='./data/test.csv',
    transform=val_transform,
    max_frames=16,
    gpu_preprocessing=True,
    cache_frames=False
)

print(f"📊 数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")

# 优化批次大小配置
# 动态调整批次大小以最大化GPU利用率
if torch.cuda.is_available():
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb >= 14:  # T4 GPU
        batch_size = 4  # 增加批次大小以提升效率
    else:
        batch_size = 2
else:
    batch_size = 1

print(f"使用批次大小: {batch_size} (基于GPU内存自动调整)")

# 优化数据加载器配置
if IS_KAGGLE:
    num_workers = 2  # Kaggle环境优化
    prefetch_factor = 4  # 增加预取因子
    persistent_workers = True
else:
    num_workers = min(4, mp.cpu_count())  # 本地环境使用更多workers
    prefetch_factor = 6
    persistent_workers = True

print(f"🔥 数据加载配置: {num_workers} workers, 预取因子: {prefetch_factor}")

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # 启用pin_memory提升传输效率
    drop_last=True,  # 确保批次大小一致
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    persistent_workers=persistent_workers if num_workers > 0 else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    persistent_workers=persistent_workers if num_workers > 0 else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    persistent_workers=persistent_workers if num_workers > 0 else False
)

print(f"✅ 数据加载器创建完成")
print(f"训练批次数: {len(train_loader)} (批次大小: {batch_size})")
print(f"验证批次数: {len(val_loader)}")
print(f"测试批次数: {len(test_loader)}")
print(f"数据加载worker数: {num_workers}")