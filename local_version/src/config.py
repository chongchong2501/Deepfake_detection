# 配置文件 - 本地RTX4070 Laptop优化版本

import os
import torch
import random
import numpy as np
from pathlib import Path

class Config:
    """项目配置类"""
    
    # 基础路径配置
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT.parent / "dataset" / "FaceForensics++_C23"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"  # 主要输出目录
    MODELS_DIR = OUTPUT_DIR / "models"     # 模型保存目录
    LOGS_DIR = OUTPUT_DIR / "logs"         # 日志目录
    RESULTS_DIR = OUTPUT_DIR / "results"   # 结果保存目录
    DATA_CACHE_DIR = OUTPUT_DIR / "data"   # 数据缓存目录
    
    # 数据配置
    DATA_DIR = DATA_ROOT  # 数据目录
    MAX_VIDEOS_PER_CLASS = 500  # 每类最大视频数
    MAX_FRAMES = 16  # 每个视频提取的最大帧数
    FRAME_SIZE = (224, 224)  # 帧尺寸
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 模型配置
    BACKBONE = 'resnet50'
    HIDDEN_DIM = 512
    NUM_LSTM_LAYERS = 2
    DROPOUT = 0.3
    USE_ATTENTION = True
    
    # 训练配置 - RTX4070优化
    BATCH_SIZE = 12  # RTX4070可以支持更大的批次
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    
    # RTX4070 Laptop优化配置
    NUM_WORKERS = 4  # 本地环境可以使用更多worker
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    
    # GPU配置
    USE_CUDA = True  # 使用CUDA
    USE_MIXED_PRECISION = True  # RTX4070支持混合精度
    GPU_MEMORY_FRACTION = 0.85  # 保守的内存使用
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 8
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Focal Loss配置
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # 随机种子
    RANDOM_SEED = 42
    
    @classmethod
    def setup_environment(cls):
        """设置环境和随机种子"""
        # 设置随机种子
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.RANDOM_SEED)
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)
            
            # RTX4070优化配置
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 设置GPU内存使用
            torch.cuda.set_per_process_memory_fraction(cls.GPU_MEMORY_FRACTION)
            
            print(f"✅ GPU配置完成: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 创建必要目录
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for directory in [cls.MODELS_DIR, cls.LOGS_DIR, cls.RESULTS_DIR, cls.DATA_CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ 环境配置完成")
    
    @classmethod
    def get_device(cls):
        """获取计算设备"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n=== 项目配置信息 ===")
        print(f"数据路径: {cls.DATA_ROOT}")
        print(f"模型: {cls.BACKBONE} + LSTM + Attention")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"设备: {cls.get_device()}")
        print(f"混合精度: {cls.USE_MIXED_PRECISION}")
        print("=====================\n")

# 全局配置实例
config = Config()