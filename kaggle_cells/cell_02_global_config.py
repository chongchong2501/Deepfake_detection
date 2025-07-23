# Cell 2: 全局配置和工具函数 - Kaggle T4 优化版本

import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Kaggle环境优化：平衡性能和可重复性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)

# Kaggle T4 GPU配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎮 GPU数量: {gpu_count}")
    print(f"🎮 GPU型号: {gpu_name}")
    print(f"💾 单GPU内存: {gpu_memory:.1f} GB")
    print(f"💾 总GPU内存: {gpu_memory * gpu_count:.1f} GB")
    
    # 多GPU配置
    USE_MULTI_GPU = gpu_count > 1
    if USE_MULTI_GPU:
        print(f"✅ 检测到 {gpu_count} 个GPU，启用多GPU并行训练")
        # 双T4 GPU优化配置
        torch.cuda.set_per_process_memory_fraction(0.8)  # 双T4可以使用更多内存
    else:
        print("📝 单GPU模式")
        torch.cuda.set_per_process_memory_fraction(0.7)  # 单GPU保守配置
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("✅ Kaggle T4 GPU优化配置已启用")
else:
    USE_MULTI_GPU = False

# 创建必要的目录
for dir_name in ['./data', './models', './logs', './results']:
    os.makedirs(dir_name, exist_ok=True)

# Kaggle环境检测
IS_KAGGLE = os.path.exists('/kaggle')
BASE_DATA_DIR = '/kaggle/input/ff-c23/FaceForensics++_C23' if IS_KAGGLE else './dataset/FaceForensics++_C23'

# 统一数据类型配置 - 全部使用FP32提升兼容性
USE_FP32_ONLY = True  # 强制使用FP32，确保最佳兼容性
print(f"数据类型策略: FP32 (兼容性优先)")

print(f"环境: {'Kaggle' if IS_KAGGLE else '本地'}")
print(f"数据基础路径: {BASE_DATA_DIR}")
print("✅ 环境设置完成")