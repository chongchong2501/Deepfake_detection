# Cell 10: 创建数据加载器 - Kaggle T4 优化版本

print("📊 创建数据加载器...")

# 简化数据变换 - 使用GPU预处理替代CPU变换
train_transform = None
val_transform = None

print(f"🔧 创建数据集（Kaggle T4优化配置）...")
print(f"📊 数据类型: FP32 (兼容性优先)")

# 优先使用重采样后的平衡数据集
balanced_train_file = './data/train_balanced.csv'
original_train_file = './data/train.csv'

if os.path.exists(balanced_train_file):
    train_csv_file = balanced_train_file
    print("🔄 使用重采样后的平衡训练数据集")
else:
    train_csv_file = original_train_file
    print("📊 使用原始训练数据集")

train_dataset = DeepfakeVideoDataset(
    csv_file=train_csv_file,
    transform=train_transform,
    max_frames=16,
    gpu_preprocessing=True,    # 启用GPU预处理
    cache_frames=False        # 禁用缓存以节省内存
)

val_dataset = DeepfakeVideoDataset(
    csv_file='./data/val.csv',
    transform=val_transform,
    max_frames=16,
    gpu_preprocessing=True,    # 启用GPU预处理
    cache_frames=False        # 禁用缓存以节省内存
)

test_dataset = DeepfakeVideoDataset(
    csv_file='./data/test.csv',
    transform=val_transform,
    max_frames=16,
    gpu_preprocessing=True,    # 启用GPU预处理
    cache_frames=False        # 禁用缓存以节省内存
)
print("✅ 数据集创建完成，已优化Kaggle T4环境配置")

# Kaggle T4 GPU批次大小优化
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU内存: {gpu_memory:.1f} GB")
    
    # 保守的批次大小设置 - 确保稳定性和兼容性
    if gpu_memory >= 15:  # T4 GPU
        batch_size = 8  # 保守设置，确保稳定
    elif gpu_memory >= 8:
        batch_size = 6
    else:
        batch_size = 4
else:
    batch_size = 4

print(f"使用批次大小: {batch_size} (Kaggle T4优化，稳定性优先)")

# Kaggle环境多进程配置 - 简化版本
if IS_KAGGLE:
    # Kaggle环境：使用单进程避免序列化问题
    num_workers = 0
    prefetch_factor = None
    persistent_workers = False
    print("📝 Kaggle环境：使用单进程模式")
else:
    # 本地环境：使用少量worker
    num_workers = 2
    prefetch_factor = 2
    persistent_workers = False
    print(f"🔥 本地环境：使用 {num_workers} workers")

print(f"数据加载配置: {num_workers} workers, 预取因子: {prefetch_factor}")

# 创建数据加载器 - Kaggle T4优化版本
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=False,  # GPU预处理，无需pin_memory
    drop_last=True,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    persistent_workers=persistent_workers if num_workers > 0 else False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=False,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    persistent_workers=persistent_workers if num_workers > 0 else False
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=False,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    persistent_workers=persistent_workers if num_workers > 0 else False
)

print(f"\n📊 数据加载器统计:")
print(f"训练批次数: {len(train_loader)} (批次大小: {batch_size})")
print(f"验证批次数: {len(val_loader)}")
print(f"测试批次数: {len(test_loader)}")
print(f"数据加载worker数: {num_workers}")
print("✅ 数据加载器创建完成")