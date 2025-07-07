# 配置文件 - 本地RTX4070 Laptop优化版本

import os
import torch
import random
import numpy as np
import yaml
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
    
    # 数据配置（默认值，可被YAML配置覆盖）
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
    def load_config(cls, config_path=None):
        """从YAML文件加载配置"""
        if config_path is None:
            config_path = cls.PROJECT_ROOT / "configs" / "default.yaml"
        
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"⚠️ 配置文件不存在: {config_path}，使用默认配置")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            print(f"📋 加载配置文件: {config_path}")
            
            # 更新数据配置
            if 'data' in yaml_config:
                data_config = yaml_config['data']
                if 'max_videos_per_class' in data_config:
                    cls.MAX_VIDEOS_PER_CLASS = data_config['max_videos_per_class']
                    print(f"  ✓ max_videos_per_class: {cls.MAX_VIDEOS_PER_CLASS}")
                if 'max_frames' in data_config:
                    cls.MAX_FRAMES = data_config['max_frames']
                    print(f"  ✓ max_frames: {cls.MAX_FRAMES}")
                if 'frame_size' in data_config:
                    cls.FRAME_SIZE = tuple(data_config['frame_size'])
                    print(f"  ✓ frame_size: {cls.FRAME_SIZE}")
                if 'train_split' in data_config:
                    cls.TRAIN_RATIO = data_config['train_split']
                if 'val_split' in data_config:
                    cls.VAL_RATIO = data_config['val_split']
                if 'test_split' in data_config:
                    cls.TEST_RATIO = data_config['test_split']
            
            # 更新训练配置
            if 'training' in yaml_config:
                training_config = yaml_config['training']
                if 'epochs' in training_config:
                    cls.NUM_EPOCHS = training_config['epochs']
                if 'batch_size' in training_config:
                    cls.BATCH_SIZE = training_config['batch_size']
                if 'learning_rate' in training_config:
                    cls.LEARNING_RATE = training_config['learning_rate']
                if 'weight_decay' in training_config:
                    cls.WEIGHT_DECAY = training_config['weight_decay']
            
            # 更新模型配置
            if 'model' in yaml_config:
                model_config = yaml_config['model']
                if 'backbone' in model_config:
                    cls.BACKBONE = model_config['backbone']
                if 'hidden_dim' in model_config:
                    cls.HIDDEN_DIM = model_config['hidden_dim']
                if 'num_layers' in model_config:
                    cls.NUM_LSTM_LAYERS = model_config['num_layers']
                if 'dropout' in model_config:
                    cls.DROPOUT = model_config['dropout']
                if 'use_attention' in model_config:
                    cls.USE_ATTENTION = model_config['use_attention']
            
            # 更新数据加载配置
            if 'dataloader' in yaml_config:
                dataloader_config = yaml_config['dataloader']
                if 'num_workers' in dataloader_config:
                    cls.NUM_WORKERS = dataloader_config['num_workers']
                if 'pin_memory' in dataloader_config:
                    cls.PIN_MEMORY = dataloader_config['pin_memory']
                if 'prefetch_factor' in dataloader_config:
                    cls.PREFETCH_FACTOR = dataloader_config['prefetch_factor']
            
            print("✅ 配置文件加载完成")
            
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            print("使用默认配置")
    
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
        print(f"每类最大视频数: {cls.MAX_VIDEOS_PER_CLASS}")
        print(f"最大帧数: {cls.MAX_FRAMES}")
        print(f"帧尺寸: {cls.FRAME_SIZE}")
        print(f"模型: {cls.BACKBONE} + LSTM + Attention")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"设备: {cls.get_device()}")
        print(f"混合精度: {cls.USE_MIXED_PRECISION}")
        print("=====================\n")

# 全局配置实例
config = Config()