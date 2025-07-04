# Kaggle深度伪造检测项目 - 独立模块

本项目将原始的Kaggle笔记本拆分为9个独立的Python模块，每个模块都可以在Kaggle环境中作为单独的代码单元格运行。

## 📁 模块列表

| 模块文件 | 功能描述 | 主要内容 |
|---------|---------|----------|
| `module_1.py` | 环境设置和导入 | 导入必要的库，设置随机种子，检查GPU，创建目录 |
| `module_2.py` | 数据下载与预处理 | 视频帧提取，数据集划分，质量检查 |
| `module_3.py` | 数据集类定义 | 高级数据集类，数据增强，MixUp/CutMix |
| `module_4.py` | 模型定义 | 深度伪造检测模型架构，损失函数 |
| `module_5.py` | 训练和验证函数 | 训练循环，验证循环，早停机制 |
| `module_6.py` | 模型训练配置 | 训练参数设置，优化器配置 |
| `module_7.py` | 执行训练循环 | 完整的训练流程，模型保存 |
| `module_8.py` | 模型评估 | 基础模型评估，指标计算 |
| `module_9.py` | 完整的模型评估和结果分析 | 详细评估，可视化，报告生成 |

## 🚀 使用方法

### 在Kaggle中使用

1. **创建新的Kaggle笔记本**
   - 确保启用GPU加速器（Tesla T4或更高）
   - 添加FaceForensics++数据集

2. **按顺序运行模块**
   ```
   module_1.py → module_2.py → module_3.py → ... → module_9.py
   ```

3. **每个模块的使用步骤**：
   - 创建新的代码单元格
   - 复制对应模块文件的全部内容
   - 粘贴到代码单元格中
   - 运行单元格

### 本地使用

1. **环境准备**
   ```bash
   pip install torch torchvision opencv-python pandas numpy matplotlib seaborn scikit-learn tqdm albumentations
   ```

2. **数据准备**
   - 下载FaceForensics++数据集
   - 调整`module_2.py`中的数据路径

3. **运行模块**
   ```bash
   python module_1.py
   python module_2.py
   # ... 依次运行所有模块
   ```

## 📊 模块详细说明

### Module 1: 环境设置
- 导入所有必要的Python库
- 设置随机种子确保结果可复现
- 检查GPU可用性
- 创建必要的目录结构

### Module 2: 数据预处理
- 自动检测Kaggle环境
- 内存友好的视频帧提取
- 智能采样和质量过滤
- 数据集划分（训练/验证/测试）

### Module 3: 数据集类
- 高级深度伪造数据集类
- 支持时序和空间数据增强
- MixUp和CutMix增强技术
- 类别权重计算

### Module 4: 模型架构
- 优化的深度伪造检测模型
- ResNet50 + LSTM + 注意力机制
- Focal Loss损失函数
- 可配置的模型参数

### Module 5: 训练函数
- 训练和验证循环
- 早停机制
- 学习率调度
- 梯度裁剪

### Module 6: 训练配置
- 训练超参数设置
- 优化器和调度器配置
- 数据加载器创建

### Module 7: 训练执行
- 完整的训练流程
- 模型检查点保存
- 训练日志记录
- 可视化训练过程

### Module 8: 基础评估
- 模型性能评估
- 基础指标计算
- 混淆矩阵生成

### Module 9: 完整评估
- 详细的模型评估
- 多种可视化图表
- 性能分析报告
- 结果保存和导出

## ⚙️ 系统要求

### Kaggle环境
- GPU加速器：Tesla T4或更高
- 内存：至少16GB RAM
- 存储：至少5GB可用空间

### 本地环境
- Python 3.8+
- CUDA 11.0+（如果使用GPU）
- 至少8GB RAM
- 至少10GB可用存储空间

## 📈 预期结果

运行完所有模块后，您将获得：

1. **训练好的模型**：`./models/best_model.pth`
2. **评估报告**：
   - `./results/evaluation/detailed_evaluation_report.json`
   - `./results/evaluation/evaluation_summary.csv`
3. **可视化图表**：
   - 混淆矩阵
   - ROC和PR曲线
   - 训练过程图表
4. **处理后的数据**：
   - `./data/train.csv`
   - `./data/val.csv`
   - `./data/test.csv`

## 🔧 故障排除

### 常见问题

1. **内存不足**
   - 减少`batch_size`
   - 减少`max_frames`参数
   - 减少`max_videos_per_class`

2. **GPU内存不足**
   - 使用更小的模型backbone
   - 减少序列长度
   - 启用梯度检查点

3. **数据加载错误**
   - 检查数据路径
   - 确保数据集格式正确
   - 检查文件权限

### 性能优化建议

1. **Kaggle环境**
   - 使用GPU加速器
   - 启用网络连接
   - 合理设置num_workers

2. **本地环境**
   - 使用SSD存储
   - 增加系统RAM
   - 使用多GPU训练

## 📝 注意事项

1. **运行顺序**：必须按照模块编号顺序运行
2. **依赖关系**：后续模块依赖前面模块的输出
3. **资源管理**：注意内存和GPU使用情况
4. **数据安全**：确保数据集使用符合相关协议

## 🤝 贡献

如果您发现问题或有改进建议，请：
1. 检查现有的issue
2. 创建详细的bug报告
3. 提供复现步骤
4. 建议解决方案

## 📄 许可证

本项目仅用于学术研究和教育目的。使用FaceForensics++数据集需要遵循其相应的许可协议。