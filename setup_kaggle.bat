@echo off
REM 视频深度伪造检测项目 - Kaggle环境设置脚本 (Windows版本)

REM 检查Python是否已安装
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python未安装，请先安装Python 3.7+
    exit /b 1
)

REM 检查pip是否已安装
pip --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo pip未安装，请先安装pip
    exit /b 1
)

REM 安装依赖
echo 安装项目依赖...
pip install -r requirements.txt

REM 检查Kaggle API凭证
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo 请确保您已配置Kaggle API凭证
    echo 请访问 https://www.kaggle.com/account 获取API令牌
    echo 然后创建目录 %USERPROFILE%\.kaggle 并将kaggle.json放入其中
    exit /b 1
)

REM 下载FaceForensics++ Dataset (C23)
echo 下载FaceForensics++ Dataset (C23)...
REM 注意：用户需要接受数据集使用条款
kaggle datasets download -d ciplab/faceforensicspp-c23-dataset

REM 解压数据集
echo 解压数据集...
powershell -Command "Expand-Archive -Path faceforensicspp-c23-dataset.zip -DestinationPath data/ -Force"

REM 创建必要的目录
if not exist "checkpoints" mkdir checkpoints
if not exist "results" mkdir results

REM 启动Jupyter Notebook
echo 启动Jupyter Notebook...
jupyter notebook deepfake_detection.ipynb

echo 设置完成！
pause