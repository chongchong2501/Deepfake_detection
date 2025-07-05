# Cell 10: 创建数据加载器

print("📊 创建数据加载器...")

# 获取数据变换
train_transform = get_transforms('train')
val_transform = get_transforms('val')

# 创建数据集
train_dataset = DeepfakeVideoDataset('./data/train.csv', transform=train_transform)
val_dataset = DeepfakeVideoDataset('./data/val.csv', transform=val_transform)
test_dataset = DeepfakeVideoDataset('./data/test.csv', transform=val_transform)

# 根据GPU内存调整批次大小
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory >= 16:
        batch_size = 16
    elif gpu_memory >= 8:
        batch_size = 8
    else:
        batch_size = 4
else:
    batch_size = 2

print(f"使用批次大小: {batch_size}")

# 创建数据加载器
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=2,
    pin_memory=torch.cuda.is_available()
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