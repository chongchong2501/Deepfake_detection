# Kaggle Cells - 深度伪造检测模块

本目录包含深度伪造检测系统的所有核心模块，按执行顺序编号。

## 📋 执行顺序

### 🔧 环境准备 (Cell 01-02)
- `cell_01_imports_and_setup.py` - 导入库和环境设置
- `cell_02_global_config.py` - 全局配置参数

### 📊 数据处理 (Cell 03-04)  
- `cell_03_data_processing.py` - 数据预处理函数
- `cell_04_dataset_class.py` - 数据集类定义

### 🧠 模型构建 (Cell 05-08)
- `cell_05_model_definition.py` - 模型架构定义
- `cell_06_loss_and_utils.py` - 损失函数和工具
- `cell_07_training_functions.py` - 训练相关函数
- `cell_08_evaluation_functions.py` - 评估相关函数

### 🚀 训练流程 (Cell 09-13)
- `cell_09_data_preparation.py` - 数据准备和加载
- `cell_10_data_loaders.py` - 数据加载器创建
- `cell_11_model_setup.py` - 模型初始化设置
- `cell_12_training_loop.py` - 主训练循环
- `cell_13_model_evaluation.py` - 模型性能评估

### 📈 结果输出 (Cell 14-15)
- `cell_14_results_summary.py` - 训练结果总结
- `cell_15_ensemble_inference.py` - 集成推理系统

## ⚡ 快速运行

### 完整训练流程
```bash
# 按顺序执行所有模块
for i in {01..14}; do
    python cell_${i}_*.py
done
```

### 仅推理预测
```bash
# 使用训练好的模型进行推理
python cell_15_ensemble_inference.py
```

## 🎯 核心功能

- **MTCNN人脸检测**: 高精度人脸区域提取
- **多模态特征**: 频域分析 + 压缩伪影检测
- **集成学习**: 多模型融合提升准确率
- **GPU优化**: 混合精度训练和推理加速
- **实时监控**: 训练过程可视化和性能监控

## 📝 使用说明

1. **首次运行**: 依次执行 cell_01 到 cell_14
2. **推理预测**: 训练完成后运行 cell_15
3. **配置调整**: 修改 cell_02 中的全局参数
4. **数据路径**: 在 cell_09 中设置数据集路径

## ⚙️ 配置要点

```python
# GPU内存优化
BATCH_SIZE = 4      # 根据GPU内存调整
USE_AMP = True      # 启用混合精度训练

# 特征提取
ENABLE_MTCNN = True     # MTCNN人脸检测
ENABLE_FREQUENCY = True  # 频域特征分析
ENABLE_ENSEMBLE = True   # 集成学习模式
```

## 🔍 故障排除

- **内存不足**: 减小BATCH_SIZE或禁用GPU预处理
- **依赖缺失**: 检查MTCNN和SciPy安装
- **训练中断**: 模型会自动保存检查点，可继续训练