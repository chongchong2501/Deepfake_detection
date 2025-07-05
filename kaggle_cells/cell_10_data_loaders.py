# Cell 10: 创建数据加载器

print("📊 创建数据加载器...")

# 获取数据变换 - 简化变换以减少CPU负担
train_transform = None  # 使用GPU预处理替代CPU变换
val_transform = None

# 创建数据集 - 启用GPU预处理
train_dataset = DeepfakeVideoDataset('./data/train.csv', transform=train_transform, max_frames=16, gpu_preprocessing=True)
val_dataset = DeepfakeVideoDataset('./data/val.csv', transform=val_transform, max_frames=16, gpu_preprocessing=True)
test_dataset = DeepfakeVideoDataset('./data/test.csv', transform=val_transform, max_frames=16, gpu_preprocessing=True)

# 优化批次大小以减少CPU瓶颈
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU，每个GPU内存: {gpu_memory:.1f} GB")
    
    # 大幅降低批次大小以减少CPU负担
    batch_size = 4  # 固定使用小批次
else:
    batch_size = 2

print(f"使用批次大小: {batch_size} (优化CPU性能)")

# 大幅降低worker数量以减少CPU瓶颈
num_workers = 2  # 固定使用2个worker
print(f"使用 {num_workers} 个数据加载worker (优化CPU性能)")

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
    persistent_workers=False,  # 禁用以减少内存占用
    prefetch_factor=1  # 降低预取因子
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,  # 禁用以减少内存占用
    prefetch_factor=1  # 降低预取因子
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,  # 禁用以减少内存占用
    prefetch_factor=1  # 降低预取因子
)

print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(val_loader)}")
print(f"测试批次数: {len(test_loader)}")

# 测试数据加载器
print("\n🔍 测试数据加载器...")
try:
    sample_batch = next(iter(train_loader))
    videos, labels = sample_batch
    print(f"视频张量形状: {videos.shape}")
    print(f"标签张量形状: {labels.shape}")
    print(f"视频数据类型: {videos.dtype}")
    print(f"标签数据类型: {labels.dtype}")
    print(f"视频数据范围: [{videos.min():.3f}, {videos.max():.3f}]")
    print(f"标签分布: {labels.unique(return_counts=True)}")
    print("✅ 数据加载器测试成功")
except Exception as e:
    print(f"❌ 数据加载器测试失败: {e}")
    raise e

print("✅ 数据加载器创建完成")