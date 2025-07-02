#!/bin/bash

# 视频深度伪造检测项目 - Kaggle环境设置脚本

# 确保Kaggle API凭证已配置
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "请确保您已配置Kaggle API凭证"
    echo "请访问 https://www.kaggle.com/account 获取API令牌"
    echo "然后运行: mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 下载FaceForensics++ Dataset (C23)
echo "下载FaceForensics++ Dataset (C23)..."
# 注意：用户需要接受数据集使用条款
kaggle datasets download -d ciplab/faceforensicspp-c23-dataset
unzip -q faceforensicspp-c23-dataset.zip -d data/

# 创建必要的目录
mkdir -p checkpoints
mkdir -p results

# 启动Jupyter Notebook
echo "启动Jupyter Notebook..."
jupyter notebook deepfake_detection.ipynb

echo "设置完成！"