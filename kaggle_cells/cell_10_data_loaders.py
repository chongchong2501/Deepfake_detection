# Cell 10: 创建数据加载器 - Kaggle T4 优化版本

import multiprocessing as mp

print("📊 创建数据加载器...")

# 简化数据变换 - 使用GPU预处理替代CPU变换
train_transform = None
val_transform = None

# 多GPU优化配置
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
is_multi_gpu = gpu_count > 1

print(f"🔧 创建数据集（Kaggle T4优化配置）...")
print(f"📊 数据类型: FP32 (兼容性优先)")

# 创建数据集 - 双T4 GPU优化配置
train_dataset = DeepfakeVideoDataset(
    csv_file='./data/train.csv',
    transform=train_transform,
    max_frames=12,  # 适中的帧数，平衡性能和内存
    gpu_preprocessing=False,  # 在多进程环境中禁用GPU预处理
    cache_frames=False,  # 避免内存压力
    extract_fourier=True,  # 重新启用频域特征提取
    extract_compression=True  # 重新启用压缩特征提取
)

val_dataset = DeepfakeVideoDataset(
    csv_file='./data/val.csv',
    transform=val_transform,
    max_frames=12,  # 适中的帧数，平衡性能和内存
    gpu_preprocessing=False,
    cache_frames=False,
    extract_fourier=True,  # 重新启用频域特征提取
    extract_compression=True  # 重新启用压缩特征提取
)

test_dataset = DeepfakeVideoDataset(
    csv_file='./data/test.csv',
    transform=val_transform,
    max_frames=12,  # 适中的帧数，平衡性能和内存
    gpu_preprocessing=False,
    cache_frames=False,
    extract_fourier=True,  # 重新启用频域特征提取
    extract_compression=True  # 重新启用压缩特征提取
)

print(f"📊 数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")

# 批次大小配置 - 根据GPU数量调整
batch_size = 2  # 基础批次大小
if is_multi_gpu:
    print(f"🚀 多GPU模式: {gpu_count} 个GPU")
    print(f"📦 单GPU批次大小: {batch_size}")
    print(f"📦 总有效批次大小: {batch_size * gpu_count}")
else:
    print(f"📝 单GPU模式")
    print(f"📦 批次大小: {batch_size}")

# 工作进程数优化
num_workers = 0  # Kaggle环境使用单进程模式确保稳定性

# 优化数据加载器配置 - 减少worker数量以避免崩溃
if IS_KAGGLE:
    prefetch_factor = None
    persistent_workers = False
else:
    prefetch_factor = None
    persistent_workers = False

print(f"🔥 数据加载配置: {num_workers} workers (单进程模式确保稳定性)")

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False,  # 在单进程模式下禁用pin_memory
    drop_last=True,  # 确保批次大小一致
    prefetch_factor=prefetch_factor,
    persistent_workers=persistent_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
    prefetch_factor=prefetch_factor,
    persistent_workers=persistent_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
    prefetch_factor=prefetch_factor,
    persistent_workers=persistent_workers
)

print(f"✅ 数据加载器创建完成")
print(f"训练批次数: {len(train_loader)} (批次大小: {batch_size})")
print(f"验证批次数: {len(val_loader)}")
print(f"测试批次数: {len(test_loader)}")
print(f"数据加载worker数: {num_workers}")
print("⚠️ 使用单进程模式确保稳定性，如需多进程请确保数据路径正确且帧提取函数可用")