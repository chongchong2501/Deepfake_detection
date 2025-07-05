# 深度伪造检测模型 - Kaggle Notebook 使用指南

本项目将深度伪造检测模型分解为14个独立的Jupyter Notebook单元，便于在Kaggle环境中运行。

## 📁 文件结构

```
kaggle_cells/
├── cell_01_imports_and_setup.py      # 库导入和环境设置
├── cell_02_global_config.py           # 全局配置和随机种子
├── cell_03_data_processing.py         # 数据处理函数
├── cell_04_dataset_class.py           # 数据集类定义
├── cell_05_model_definition.py        # 模型架构定义
├── cell_06_loss_and_utils.py          # 损失函数和工具类
├── cell_07_training_functions.py      # 训练和验证函数
├── cell_08_evaluation_functions.py    # 评估和可视化函数
├── cell_09_data_preparation.py        # 数据准备和预处理
├── cell_10_data_loaders.py            # 数据加载器创建
├── cell_11_model_setup.py             # 模型初始化和配置
├── cell_12_training_loop.py           # 模型训练主循环
├── cell_13_model_evaluation.py        # 模型评估和结果分析
├── cell_14_results_summary.py         # 结果保存和总结
└── README.md                          # 本说明文件
```

## 🚀 在Kaggle中的使用步骤

### 1. 创建新的Kaggle Notebook
- 登录Kaggle并创建新的Notebook
- 选择GPU加速器（推荐使用GPU P100或T4）
- 确保开启互联网访问

### 2. 数据准备
在第一个cell之前，添加数据集：
```python
# 如果使用Kaggle数据集，添加数据集到notebook
# 或者上传自己的数据集
# 确保数据路径正确设置
```

### 3. 按顺序执行Cell

#### Cell 1: 导入和设置
```python
# 复制 cell_01_imports_and_setup.py 的内容
# 安装必要的库（如果需要）
```

#### Cell 2: 全局配置
```python
# 复制 cell_02_global_config.py 的内容
# 设置随机种子和环境
```

#### Cell 3-8: 函数定义
```python
# 依次复制 cell_03 到 cell_08 的内容
# 这些cell定义了所有必要的函数和类
```

#### Cell 9: 数据准备
```python
# 复制 cell_09_data_preparation.py 的内容
# 根据你的数据集路径修改 base_path
base_path = "/kaggle/input/your-dataset-name/"
```

#### Cell 10-14: 训练和评估
```python
# 依次复制并执行剩余的cell
# 这些cell执行实际的训练和评估过程
```

## ⚙️ 重要配置说明

### 数据路径配置
在 `cell_02_global_config.py` 中修改数据路径：
```python
# Kaggle环境检测和路径设置
if os.path.exists('/kaggle/input'):
    print("检测到Kaggle环境")
    # 修改为你的数据集路径
    base_path = "/kaggle/input/your-deepfake-dataset/"
else:
    print("本地环境")
    base_path = "./data/"
```

### 模型配置优化
为了适应Kaggle环境，已进行以下优化：
- 使用ResNet18作为轻量级backbone
- 动态调整批次大小以适应GPU内存
- 启用混合精度训练以节省内存
- 设置合理的训练轮数（15轮）

### 内存优化设置
```python
# 在训练过程中会自动进行内存清理
torch.cuda.empty_cache()
gc.collect()
```

## 📊 预期输出

运行完成后，将生成以下文件和结果：

### 模型文件
- `./models/best_model.pth` - 最佳模型权重

### 结果文件
- `./results/experiment_results.json` - 详细实验结果
- `./results/experiment_report.txt` - 实验报告
- `./results/training_history.csv` - 训练历史数据
- `./results/test_predictions.csv` - 测试预测结果

### 可视化图表
- `./results/training_history.png` - 训练历史曲线
- `./results/evaluation/confusion_matrix.png` - 混淆矩阵
- `./results/evaluation/roc_pr_curves.png` - ROC和PR曲线
- `./results/evaluation/score_distribution.png` - 预测分数分布

## 🔧 故障排除

### 常见问题

1. **内存不足错误**
   - 减小批次大小
   - 使用更小的模型backbone
   - 启用梯度检查点

2. **数据路径错误**
   - 检查数据集是否正确添加到notebook
   - 确认路径设置正确

3. **CUDA错误**
   - 确保选择了GPU加速器
   - 检查CUDA兼容性

### 性能优化建议

1. **使用GPU加速器**
   - 推荐使用GPU P100或T4
   - 启用混合精度训练

2. **数据加载优化**
   - 适当设置num_workers
   - 使用pin_memory=True

3. **模型配置调整**
   - 根据数据集大小调整模型复杂度
   - 使用适当的学习率和批次大小

## 📝 使用注意事项

1. **按顺序执行**: 必须按照cell的编号顺序执行，因为后续cell依赖前面定义的函数和变量

2. **路径设置**: 确保在cell_02中正确设置数据路径

3. **资源监控**: 在Kaggle中注意GPU使用时间限制

4. **结果保存**: 重要结果会自动保存，可以下载查看

5. **实验记录**: 所有实验参数和结果都会记录在JSON文件中

## 🎯 预期性能

在标准深度伪造数据集上，该模型预期达到：
- 准确率: 85-95%
- AUC-ROC: 0.90-0.98
- F1分数: 0.85-0.95

具体性能取决于数据集质量和大小。

## 📞 支持

如果在使用过程中遇到问题，请检查：
1. 是否按顺序执行了所有cell
2. 数据路径是否正确设置
3. GPU内存是否充足
4. 所有依赖库是否正确安装

祝您使用愉快！🚀