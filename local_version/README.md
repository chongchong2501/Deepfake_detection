# 深度伪造检测系统 - 本地RTX4070版本

一个基于深度学习的视频深度伪造检测系统，专门针对本地RTX4070 Laptop环境优化。该系统使用ResNet50+双向LSTM+多头自注意力机制的先进架构，能够高效准确地检测视频中的深度伪造内容。

## 🚀 主要特性

- **先进的模型架构**: ResNet50骨干网络 + 双向LSTM + 多头自注意力机制
- **RTX4070优化**: 专门针对RTX4070 Laptop GPU进行性能优化
- **混合精度训练**: 支持FP16混合精度训练，提升训练速度和内存效率
- **GPU加速处理**: 全流程GPU加速，包括视频帧提取和预处理
- **智能缓存系统**: 优化的数据缓存机制，减少重复计算
- **完整的训练流程**: 从数据预处理到模型评估的完整pipeline
- **实时推理**: 支持单个视频和批量视频的快速推理
- **可视化分析**: 丰富的训练监控和结果可视化

## 📋 系统要求

### 硬件要求
- **CPU**: AMD R9 7940H 或同等性能处理器
- **GPU**: NVIDIA RTX4070 Laptop (8GB VRAM) 或更高
- **内存**: 16GB RAM 或更多
- **存储**: 至少50GB可用空间

### 软件要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8 或 12.x
- **cuDNN**: 对应CUDA版本的cuDNN

## 🚀 快速开始

### 环境检查和安装

```bash
# 检查环境（推荐先运行）
python setup.py --check-only

# 自动安装依赖
python setup.py --install-deps
```

### 快速演示

```bash
# 5分钟快速演示（使用少量数据）
python examples/demo.py --quick

# 完整演示
python examples/demo.py
```

### 快速训练

```bash
# 演示训练（5分钟）
python train.py --config configs/quick_demo.yaml --epochs 2 --batch-size 4

# 标准训练
python train.py --epochs 20 --batch-size 8
```

## 🛠️ 详细安装指南

### 1. 克隆项目

```bash
git clone <repository-url>
cd local_version
```

### 2. 创建虚拟环境
```bash
# 使用conda (推荐)
conda create -n deepfake-detection python=3.10
conda activate deepfake-detection

# 或使用venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖
```bash
# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

## 📊 数据准备

### 数据集结构
确保你的FaceForensics++数据集按以下结构组织：
```
e:\program\Deepfake\dataset\FaceForensics++_C23\
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── fake/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

### 数据预处理
系统会自动处理视频数据，包括：
- 帧提取和质量检测
- 尺寸标准化 (224x224)
- 数据增强
- 训练/验证/测试集划分 (70%/15%/15%)

## 🎯 使用指南

### 训练模型

#### 基础训练
```bash
python train.py --data-dir "e:\program\Deepfake\dataset\FaceForensics++_C23" --output-dir "./outputs"
```

#### 自定义训练参数
```bash
python train.py \
    --data-dir "e:\program\Deepfake\dataset\FaceForensics++_C23" \
    --output-dir "./outputs" \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.001
```

#### 恢复训练
```bash
python train.py \
    --data-dir "e:\program\Deepfake\dataset\FaceForensics++_C23" \
    --output-dir "./outputs" \
    --resume "./outputs/models/best_model.pth"
```

#### 调试模式
```bash
python train.py \
    --data-dir "e:\program\Deepfake\dataset\FaceForensics++_C23" \
    --output-dir "./outputs" \
    --debug
```

### 模型评估

#### 完整评估
```bash
python evaluate.py \
    --model "./outputs/models/best_model.pth" \
    --data-dir "e:\program\Deepfake\dataset\FaceForensics++_C23" \
    --output-dir "./evaluation_results" \
    --plot-results
```

#### 快速评估
```bash
python evaluate.py \
    --model "./outputs/models/best_model.pth" \
    --data-dir "e:\program\Deepfake\dataset\FaceForensics++_C23"
```

### 视频推理

#### 单个视频推理
```bash
python inference.py \
    --model "./outputs/models/best_model.pth" \
    --video "path/to/video.mp4" \
    --output "result.json"
```

#### 批量视频推理
```bash
python inference.py \
    --model "./outputs/models/best_model.pth" \
    --video-dir "path/to/videos/" \
    --output "batch_results.csv" \
    --verbose
```

#### 自定义阈值
```bash
python inference.py \
    --model "./outputs/models/best_model.pth" \
    --video "path/to/video.mp4" \
    --threshold 0.7
```

## ⚙️ 配置说明

### 主要配置参数 (src/config.py)

```python
# 数据配置
MAX_FRAMES = 16          # 每个视频最大帧数
FRAME_SIZE = (224, 224)  # 帧尺寸
TRAIN_SPLIT = 0.7        # 训练集比例
VAL_SPLIT = 0.15         # 验证集比例

# 模型配置
BACKBONE = 'resnet50'    # 骨干网络
HIDDEN_DIM = 512         # 隐藏层维度
NUM_LSTM_LAYERS = 2      # LSTM层数
USE_ATTENTION = True     # 是否使用注意力机制

# 训练配置
BATCH_SIZE = 8           # 批次大小
NUM_EPOCHS = 30          # 训练轮数
LEARNING_RATE = 0.001    # 学习率
WEIGHT_DECAY = 1e-4      # 权重衰减

# GPU优化
USE_MIXED_PRECISION = True  # 混合精度训练
NUM_WORKERS = 4             # 数据加载器工作进程数
PIN_MEMORY = True           # 固定内存
```

### RTX4070优化配置
- **批次大小**: 8 (平衡性能和内存使用)
- **工作进程**: 4 (匹配CPU核心数)
- **混合精度**: 启用 (提升训练速度)
- **GPU内存使用**: 90% (最大化GPU利用率)

## 📈 性能指标

### 预期性能 (RTX4070 Laptop)
- **训练速度**: ~2-3分钟/epoch (FaceForensics++ C23)
- **推理速度**: ~50-100ms/视频
- **内存使用**: ~6-7GB GPU内存
- **准确率**: >95% (在FaceForensics++ C23上)
- **AUC-ROC**: >0.98

### 模型架构
```
OptimizedDeepfakeDetector(
  (backbone): ResNet50 (预训练)
  (lstm): LSTM(2048 -> 512, 2层, 双向)
  (attention): MultiheadAttention(512, 8头)
  (classifier): Sequential(
    Linear(512 -> 256)
    ReLU + Dropout(0.3)
    Linear(256 -> 128)
    ReLU + Dropout(0.3)
    Linear(128 -> 1)
  )
)
```

## 📁 项目结构

```
local_version/
├── src/                     # 核心源代码
│   ├── __init__.py         # 包初始化
│   ├── config.py           # 配置管理
│   ├── data_processing.py  # 数据处理
│   ├── dataset.py          # 数据集类
│   ├── model.py            # 模型定义
│   ├── training.py         # 训练和评估
│   └── utils.py            # 工具函数
├── examples/               # 示例和演示
│   └── demo.py            # 演示脚本
├── configs/               # 配置文件
│   ├── default.yaml       # 默认配置
│   ├── high_performance.yaml # 高性能配置
│   └── quick_demo.yaml    # 快速演示配置
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── inference.py           # 推理脚本
├── setup.py               # 环境设置
├── requirements.txt       # 依赖列表
└── README.md             # 说明文档
```

## 🔧 故障排除

### 常见问题

#### 1. CUDA内存不足
```bash
# 减少批次大小
python train.py --batch-size 4

# 或禁用混合精度
# 在config.py中设置 USE_MIXED_PRECISION = False
```

#### 2. 数据加载慢
```bash
# 减少工作进程数
# 在config.py中设置 NUM_WORKERS = 2
```

#### 3. 视频解码错误
```bash
# 安装额外的编解码器
pip install opencv-python-headless
# 或
conda install -c conda-forge ffmpeg
```

#### 4. 模型加载失败
```bash
# 检查模型文件完整性
python -c "import torch; print(torch.load('model.pth', map_location='cpu').keys())"
```

### 性能优化建议

1. **数据预处理优化**
   - 使用SSD存储数据集
   - 启用数据缓存
   - 调整NUM_WORKERS

2. **训练优化**
   - 使用混合精度训练
   - 启用梯度累积
   - 调整学习率调度

3. **推理优化**
   - 使用TensorRT优化
   - 批量推理
   - 模型量化

## 📊 监控和日志

### 训练监控
- 实时损失和准确率显示
- GPU内存使用监控
- 训练历史保存
- 早停机制

### 输出文件
```
outputs/
├── models/
│   ├── best_model.pth      # 最佳模型
│   └── final_model.pth     # 最终模型
├── logs/
│   └── training.log        # 训练日志
├── plots/
│   ├── training_curves.png # 训练曲线
│   ├── confusion_matrix.png# 混淆矩阵
│   └── roc_curves.png      # ROC曲线
└── results/
    ├── config.json         # 训练配置
    ├── training_history.json# 训练历史
    └── evaluation_metrics.json# 评估指标
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- FaceForensics++ 数据集提供者
- PyTorch 团队
- OpenCV 社区
- 所有贡献者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 Issue
- 发送邮件
- 讨论区交流

---

**注意**: 本系统仅用于研究和教育目的。请遵守相关法律法规，不要将其用于恶意目的。