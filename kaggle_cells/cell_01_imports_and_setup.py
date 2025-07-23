# Cell 1: 导入库和环境设置

# 抑制CUDA和TensorFlow警告信息
import os
import warnings

# 设置环境变量 - 必须在导入TensorFlow之前设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制所有TensorFlow日志 (0=全部, 1=INFO, 2=WARNING, 3=ERROR)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步CUDA执行
os.environ['PYTHONWARNINGS'] = 'ignore'   # 抑制Python警告

# 抑制CUDA相关警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化警告
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU内存动态增长
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 明确指定可见的GPU设备

# 抑制cuDNN/cuFFT/cuBLAS重复注册警告
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'

# 抑制所有警告
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 修复CUDA多进程问题
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 如果已经设置过，忽略错误

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import warnings
import gc
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image
warnings.filterwarnings('ignore')

# PyTorch相关
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.io import read_video
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# 初始化CUDA并抑制警告
if torch.cuda.is_available():
    # 初始化CUDA上下文以避免后续警告
    torch.cuda.init()
    # 设置CUDA设备
    torch.cuda.set_device(0)
    # 清理CUDA缓存
    torch.cuda.empty_cache()
    print(f"🚀 CUDA已初始化，检测到 {torch.cuda.device_count()} 个GPU设备")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    print("⚠️ CUDA不可用，将使用CPU模式")

# 机器学习指标
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split

# 系统监控和性能分析
import psutil
import traceback

# 高精度人脸检测 - MTCNN
try:
    # 在导入MTCNN之前进一步抑制TensorFlow警告
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # 抑制absl日志
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass
    
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
    print("✅ MTCNN已安装，支持高精度人脸检测")
    print("   - 人脸检测精度: 高")
    print("   - 检测置信度阈值: 0.9")
    print("   - API版本: 新版本 (v1.0.0+)")
    print("   - TensorFlow警告已抑制")
except ImportError:
    MTCNN_AVAILABLE = False
    print("⚠️ MTCNN未安装，将使用OpenCV人脸检测")
    print("   - 人脸检测精度: 中等")
    print("   - 建议安装MTCNN以获得更高精度:")
    print("   - 安装命令: !pip install mtcnn")
    print("   - 或者: !pip install mtcnn[tensorflow]")
    print("   - 注意: 需要TensorFlow >= 2.12")
    print("   - 影响: 人脸检测精度略有降低，但不影响整体训练")

# 视频处理 (PyAV)
try:
    import av
    PYAV_AVAILABLE = True
    print("✅ PyAV已安装，支持GPU视频处理")
except ImportError:
    PYAV_AVAILABLE = False
    print("⚠️ PyAV未安装，视频处理将回退到CPU模式")

# 数据增强
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("警告: albumentations未安装，将使用基础数据增强")

# 频域分析支持
try:
    from scipy import fftpack
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
    print("✅ SciPy已安装，支持频域分析")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy未安装，频域分析功能受限")

print("✅ 所有库导入完成")