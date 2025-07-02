# 视频深度伪造检测项目

## 项目概述
本项目旨在使用深度学习技术检测视频中的深度伪造内容，基于FaceForensics++ Dataset (C23)数据集，采用CNN和RNN混合模型架构。

## 数据集
- **FaceForensics++ Dataset (C23)**：包含真实和伪造的面部视频

## 模型架构
- 混合CNN-RNN架构
- CNN部分用于提取视频帧的空间特征
- RNN部分用于捕获时序信息

## 项目结构
```
├── data_preprocessing.py  # 数据预处理脚本
├── model.py               # 模型定义
├── train.py               # 训练脚本
├── utils.py               # 工具函数
├── evaluate.py            # 评估脚本
└── inference.py           # 推理脚本
```

## 环境要求
- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Kaggle API

## 使用方法
1. 设置Kaggle API凭证
2. 运行数据预处理脚本：`python data_preprocessing.py`
3. 训练模型：`python train.py`
4. 评估模型：`python evaluate.py`
5. 进行推理：`python inference.py --video_path [视频路径]`

## 性能指标
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数
- AUC-ROC曲线