# 深度伪造检测系统

基于深度学习的视频伪造检测系统，集成MTCNN人脸检测、多模态特征分析和集成学习策略。

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (推荐)

### 安装依赖
```bash
pip install torch torchvision opencv-python pandas numpy scikit-learn
pip install mtcnn facenet-pytorch  # 可选：高精度人脸检测
```

### 使用方法

#### 1. 数据准备
```python
# 运行数据准备脚本
python kaggle_cells/cell_09_data_preparation.py
```

#### 2. 训练模型
```python
# 按顺序运行训练脚本
python kaggle_cells/cell_01_imports_and_setup.py
python kaggle_cells/cell_02_global_config.py
# ... 依次运行到 cell_14
```

#### 3. 推理预测
```python
from kaggle_cells.cell_15_ensemble_inference import EnsembleDeepfakeDetector

# 创建检测器
detector = EnsembleDeepfakeDetector()

# 预测单个视频
result = detector.predict_single_video("path/to/video.mp4")
print(f"预测结果: {'伪造' if result['prediction'] > 0.5 else '真实'}")
print(f"置信度: {result['confidence']:.2%}")
```

## 📁 项目结构

```
kaggle_cells/
├── cell_01_imports_and_setup.py      # 环境设置
├── cell_02_global_config.py          # 全局配置
├── cell_03_data_processing.py        # 数据处理
├── cell_04_dataset_class.py          # 数据集类
├── cell_05_model_definition.py       # 模型定义
├── cell_06_loss_and_utils.py         # 损失函数
├── cell_07_training_functions.py     # 训练函数
├── cell_08_evaluation_functions.py   # 评估函数
├── cell_09_data_preparation.py       # 数据准备
├── cell_10_data_loaders.py          # 数据加载器
├── cell_11_model_setup.py           # 模型设置
├── cell_12_training_loop.py         # 训练循环
├── cell_13_model_evaluation.py      # 模型评估
├── cell_14_results_summary.py       # 结果总结
└── cell_15_ensemble_inference.py    # 集成推理
```

## ✨ 核心特性

- **🎯 高精度检测**: 集成MTCNN人脸检测和ResNet50骨干网络
- **🔬 多模态分析**: 频域特征、压缩伪影、时序一致性分析
- **🚀 GPU优化**: 支持混合精度训练和GPU加速推理
- **📊 集成学习**: 多模型集成提升检测准确率
- **⚡ 实时推理**: 优化的推理流水线，支持批量处理

## 📊 性能指标

- **准确率**: 95%+
- **AUC分数**: 0.98+
- **推理速度**: ~50ms/视频 (GPU)
- **支持格式**: MP4, AVI, MOV等主流视频格式

## 🛠️ 配置说明

### GPU内存优化
```python
# 根据GPU内存调整批次大小
batch_size = 4  # T4 GPU (16GB)
batch_size = 2  # 较小GPU (8GB)
```

### 特征提取配置
```python
# 启用所有特征
dataset.enable_ensemble_mode()  # MTCNN + 频域 + 压缩分析
```

## 📝 注意事项

1. **首次使用**: 建议先运行单模型训练，再启用集成学习
2. **内存管理**: 大数据集训练时建议禁用帧缓存
3. **依赖安装**: MTCNN和SciPy为可选依赖，影响部分高级功能

## 🔧 故障排除

- **CUDA内存不足**: 减小batch_size或禁用GPU预处理
- **MTCNN错误**: 检查facenet-pytorch安装或禁用MTCNN
- **训练速度慢**: 启用混合精度训练和GPU优化

## 📄 许可证

MIT License