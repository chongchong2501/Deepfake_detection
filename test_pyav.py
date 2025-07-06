#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PyAV安装和GPU视频处理功能
"""

import sys
import os
sys.path.append('kaggle_cells')

# 测试PyAV导入
try:
    import av
    PYAV_AVAILABLE = True
    print("✅ PyAV已安装，版本:", av.__version__)
except ImportError:
    PYAV_AVAILABLE = False
    print("❌ PyAV未安装")

# 测试torch和torchvision
try:
    import torch
    import torchvision
    from torchvision.io import read_video
    print("✅ PyTorch已安装，版本:", torch.__version__)
    print("✅ TorchVision已安装，版本:", torchvision.__version__)
    print("✅ CUDA可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("   GPU设备:", torch.cuda.get_device_name(0))
except ImportError as e:
    print("❌ PyTorch导入失败:", e)

# 测试其他关键库
libraries = [
    ('numpy', 'np'),
    ('cv2', 'cv2'),
    ('pandas', 'pd'),
    ('matplotlib.pyplot', 'plt'),
    ('tqdm', 'tqdm'),
    ('psutil', 'psutil')
]

print("\n📦 检查其他依赖库:")
for lib_name, import_name in libraries:
    try:
        exec(f"import {import_name}")
        print(f"✅ {lib_name}")
    except ImportError:
        print(f"❌ {lib_name}")

# 测试albumentations（可选）
try:
    import albumentations as A
    print("✅ albumentations (可选增强库)")
except ImportError:
    print("⚠️ albumentations 未安装 (可选)")

print("\n🎯 总结:")
print(f"PyAV可用: {PYAV_AVAILABLE}")
torch_available = 'torch' in locals()
print(f"PyTorch可用: {torch_available}")
if torch_available:
    print(f"GPU加速: {torch.cuda.is_available()}")
else:
    print("GPU加速: 未知 (PyTorch未安装)")

if PYAV_AVAILABLE and torch_available and torch.cuda.is_available():
    print("🚀 系统已准备好进行GPU加速视频处理!")
elif PYAV_AVAILABLE and torch_available:
    print("⚡ 系统支持PyAV视频处理，但GPU不可用")
elif PYAV_AVAILABLE:
    print("⚠️ PyAV已安装，但需要PyTorch支持")
else:
    print("⚠️ 需要安装PyAV以获得最佳性能")
    print("   安装命令: pip install av")

if not torch_available:
    print("\n💡 提示: 请确保在正确的虚拟环境中运行")
    print("   激活虚拟环境: .venv\\Scripts\\activate")