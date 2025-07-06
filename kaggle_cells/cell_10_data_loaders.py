# Cell 10: 创建数据加载器

# 修复CUDA多进程问题
import multiprocessing as mp
if torch.cuda.is_available():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过，忽略错误

print("📊 创建数据加载器...")

# 获取数据变换 - 简化变换以减少CPU负担
train_transform = None  # 使用GPU预处理替代CPU变换
val_transform = None

# 创建数据集 - 启用GPU预处理和帧缓存
print("🔧 创建数据集（启用GPU预处理和帧缓存）...")
train_dataset = DeepfakeVideoDataset('./data/train.csv', transform=train_transform, max_frames=16, gpu_preprocessing=True, cache_frames=True)
val_dataset = DeepfakeVideoDataset('./data/val.csv', transform=val_transform, max_frames=16, gpu_preprocessing=True, cache_frames=True)
test_dataset = DeepfakeVideoDataset('./data/test.csv', transform=val_transform, max_frames=16, gpu_preprocessing=True, cache_frames=True)
print("✅ 数据集创建完成，已启用帧缓存以提升性能")

# 优化批次大小和数据加载性能
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU，每个GPU内存: {gpu_memory:.1f} GB")
    
    # 根据GPU内存动态调整批次大小
    if gpu_memory >= 15:  # T4或更好的GPU
        batch_size = 8
    elif gpu_memory >= 8:
        batch_size = 6
    else:
        batch_size = 4
else:
    batch_size = 2

print(f"使用批次大小: {batch_size} (根据GPU内存优化)")

# 修复多进程序列化问题
# 在Jupyter notebook环境中，自定义类无法被pickle序列化到worker进程
# 因此暂时禁用多进程，使用单进程加载
print("⚠️ 检测到Jupyter环境，为避免序列化问题，使用单进程数据加载")
num_workers = 0  # 禁用多进程以避免序列化问题
prefetch_factor = None  # 单进程时不需要预取
pin_memory = True if torch.cuda.is_available() else False
persistent_workers = False  # 单进程时不需要持久化worker
print(f"使用 {num_workers} 个数据加载worker (单进程模式)")

# 创建数据加载器（单进程模式）
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=False,  # 禁用pin_memory，因为数据已在GPU上
    drop_last=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=False  # 禁用pin_memory，因为数据已在GPU上
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=False  # 禁用pin_memory，因为数据已在GPU上
)

print(f"\n📊 数据加载器统计:")
print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(val_loader)}")
print(f"测试批次数: {len(test_loader)}")
print(f"数据加载worker数: {num_workers} (单进程模式)")
print(f"内存固定: 已禁用 (数据已在GPU上)")
print(f"帧缓存: {'启用' if train_dataset.cache_frames else '禁用'}")

# 测试数据加载器
print("\n🔍 测试数据加载器...")
import time
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