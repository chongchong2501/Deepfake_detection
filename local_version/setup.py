#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度伪造检测系统安装脚本 - 本地RTX4070版本

这个脚本帮助用户快速设置和验证深度伪造检测系统的运行环境。

使用方法:
    python setup.py [--check-only] [--install-deps]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import importlib.util

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度伪造检测系统环境设置')
    parser.add_argument('--check-only', action='store_true', 
                       help='仅检查环境，不安装依赖')
    parser.add_argument('--install-deps', action='store_true',
                       help='自动安装缺失的依赖')
    parser.add_argument('--cuda-version', type=str, default='auto',
                       choices=['auto', '11.8', '12.1'],
                       help='指定CUDA版本 (默认: auto)')
    
    return parser.parse_args()

def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"🔧 {title}")
    print("="*60)

def print_step(step, description):
    """打印步骤"""
    print(f"\n📋 {step}: {description}")
    print("-" * 40)

def check_python_version():
    """检查Python版本"""
    print_step("步骤 1", "检查Python版本")
    
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8 or version.minor > 11:
        print("❌ Python版本不兼容")
        print("   要求: Python 3.8 - 3.11")
        print(f"   当前: Python {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print("✅ Python版本兼容")
        return True

def check_cuda_availability():
    """检查CUDA可用性"""
    print_step("步骤 2", "检查CUDA环境")
    
    try:
        # 检查nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA驱动已安装")
            
            # 提取CUDA版本信息
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"   支持的CUDA版本: {cuda_version}")
                    break
            
            return True
        else:
            print("❌ NVIDIA驱动未安装或不可用")
            return False
    
    except FileNotFoundError:
        print("❌ nvidia-smi 命令未找到")
        print("   请确保已安装NVIDIA驱动")
        return False
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")
        return False

def check_package(package_name, import_name=None, version_attr=None):
    """检查Python包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, None
        
        module = importlib.import_module(import_name)
        
        if version_attr:
            version = getattr(module, version_attr, 'Unknown')
        else:
            version = getattr(module, '__version__', 'Unknown')
        
        return True, version
    
    except ImportError:
        return False, None
    except Exception:
        return False, None

def check_dependencies():
    """检查依赖包"""
    print_step("步骤 3", "检查Python依赖")
    
    # 核心依赖
    core_deps = [
        ('torch', 'torch', '__version__'),
        ('torchvision', 'torchvision', '__version__'),
        ('opencv-python', 'cv2', '__version__'),
        ('numpy', 'numpy', '__version__'),
        ('pandas', 'pandas', '__version__'),
        ('scikit-learn', 'sklearn', '__version__'),
        ('matplotlib', 'matplotlib', '__version__'),
        ('seaborn', 'seaborn', '__version__'),
        ('tqdm', 'tqdm', '__version__'),
        ('pillow', 'PIL', '__version__'),
    ]
    
    # 可选依赖
    optional_deps = [
        ('av', 'av', '__version__'),
        ('albumentations', 'albumentations', '__version__'),
        ('psutil', 'psutil', '__version__'),
    ]
    
    missing_core = []
    missing_optional = []
    
    print("核心依赖:")
    for package_name, import_name, version_attr in core_deps:
        installed, version = check_package(package_name, import_name, version_attr)
        if installed:
            print(f"  ✅ {package_name}: {version}")
        else:
            print(f"  ❌ {package_name}: 未安装")
            missing_core.append(package_name)
    
    print("\n可选依赖:")
    for package_name, import_name, version_attr in optional_deps:
        installed, version = check_package(package_name, import_name, version_attr)
        if installed:
            print(f"  ✅ {package_name}: {version}")
        else:
            print(f"  ⚠️ {package_name}: 未安装 (可选)")
            missing_optional.append(package_name)
    
    return missing_core, missing_optional

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    print_step("步骤 4", "检查PyTorch CUDA支持")
    
    try:
        import torch
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            return True
        else:
            print("❌ PyTorch CUDA支持不可用")
            return False
    
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    except Exception as e:
        print(f"❌ PyTorch CUDA检查失败: {e}")
        return False

def test_gpu_performance():
    """测试GPU性能"""
    print_step("步骤 5", "GPU性能测试")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，跳过GPU测试")
            return
        
        device = torch.device('cuda')
        print(f"使用设备: {device}")
        
        # 简单的矩阵乘法测试
        print("进行简单的GPU计算测试...")
        
        size = 1000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # 预热
        for _ in range(5):
            _ = torch.mm(a, b)
        
        torch.cuda.synchronize()
        
        # 计时测试
        import time
        start_time = time.time()
        
        for _ in range(10):
            c = torch.mm(a, b)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"✅ GPU计算测试完成")
        print(f"   矩阵大小: {size}x{size}")
        print(f"   平均计算时间: {avg_time*1000:.2f} ms")
        
        # 内存测试
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
        
    except Exception as e:
        print(f"❌ GPU性能测试失败: {e}")

def install_dependencies(missing_core, missing_optional, cuda_version='auto'):
    """安装缺失的依赖"""
    print_step("安装依赖", "自动安装缺失的包")
    
    if not missing_core and not missing_optional:
        print("✅ 所有依赖都已安装")
        return True
    
    try:
        # 安装核心依赖
        if missing_core:
            print("安装核心依赖...")
            
            # 特殊处理PyTorch
            if 'torch' in missing_core or 'torchvision' in missing_core:
                print("安装PyTorch...")
                
                if cuda_version == 'auto':
                    # 尝试检测CUDA版本
                    try:
                        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                        if 'CUDA Version: 12.' in result.stdout:
                            cuda_version = '12.1'
                        elif 'CUDA Version: 11.' in result.stdout:
                            cuda_version = '11.8'
                        else:
                            cuda_version = '11.8'  # 默认
                    except:
                        cuda_version = '11.8'  # 默认
                
                if cuda_version == '12.1':
                    torch_cmd = [
                        sys.executable, '-m', 'pip', 'install', 
                        'torch', 'torchvision', 'torchaudio',
                        '--index-url', 'https://download.pytorch.org/whl/cu121'
                    ]
                else:  # 11.8
                    torch_cmd = [
                        sys.executable, '-m', 'pip', 'install', 
                        'torch', 'torchvision', 'torchaudio',
                        '--index-url', 'https://download.pytorch.org/whl/cu118'
                    ]
                
                print(f"执行: {' '.join(torch_cmd)}")
                result = subprocess.run(torch_cmd)
                
                if result.returncode != 0:
                    print("❌ PyTorch安装失败")
                    return False
                
                # 从缺失列表中移除
                missing_core = [pkg for pkg in missing_core if pkg not in ['torch', 'torchvision']]
            
            # 安装其他核心依赖
            if missing_core:
                cmd = [sys.executable, '-m', 'pip', 'install'] + missing_core
                print(f"执行: {' '.join(cmd)}")
                result = subprocess.run(cmd)
                
                if result.returncode != 0:
                    print("❌ 核心依赖安装失败")
                    return False
        
        # 安装可选依赖
        if missing_optional:
            print("安装可选依赖...")
            cmd = [sys.executable, '-m', 'pip', 'install'] + missing_optional
            print(f"执行: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print("⚠️ 部分可选依赖安装失败（不影响核心功能）")
        
        print("✅ 依赖安装完成")
        return True
        
    except Exception as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def check_project_structure():
    """检查项目结构"""
    print_step("步骤 6", "检查项目结构")
    
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/data_processing.py',
        'src/dataset.py',
        'src/model.py',
        'src/training.py',
        'src/utils.py',
        'train.py',
        'evaluate.py',
        'inference.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ 缺失 {len(missing_files)} 个文件")
        return False
    else:
        print("\n✅ 项目结构完整")
        return True

def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("🚀 深度伪造检测系统环境设置")
    print("="*60)
    print(f"模式: {'仅检查' if args.check_only else '检查并安装'}")
    print(f"CUDA版本: {args.cuda_version}")
    print("="*60)
    
    success = True
    
    # 1. 检查Python版本
    if not check_python_version():
        success = False
    
    # 2. 检查CUDA
    cuda_available = check_cuda_availability()
    if not cuda_available:
        print("⚠️ CUDA不可用，将使用CPU模式（性能较低）")
    
    # 3. 检查依赖
    missing_core, missing_optional = check_dependencies()
    
    if missing_core:
        print(f"\n❌ 缺失核心依赖: {', '.join(missing_core)}")
        success = False
    
    # 4. 检查PyTorch CUDA
    if cuda_available:
        pytorch_cuda = check_pytorch_cuda()
        if not pytorch_cuda:
            print("⚠️ PyTorch CUDA支持不可用")
    
    # 5. GPU性能测试
    if cuda_available:
        test_gpu_performance()
    
    # 6. 检查项目结构
    if not check_project_structure():
        success = False
    
    # 安装依赖（如果需要）
    if not args.check_only and (missing_core or missing_optional):
        if install_dependencies(missing_core, missing_optional, args.cuda_version):
            print("\n🔄 重新检查依赖...")
            missing_core, missing_optional = check_dependencies()
            if not missing_core:
                success = True
    
    # 最终结果
    print_section("设置结果")
    
    if success and not missing_core:
        print("🎉 环境设置成功！")
        print("\n📚 接下来你可以：")
        print("   1. 运行 python demo.py --quick 进行快速演示")
        print("   2. 运行 python train.py 开始训练")
        print("   3. 查看 README.md 了解详细使用方法")
        return 0
    else:
        print("❌ 环境设置未完成")
        if missing_core:
            print(f"   缺失核心依赖: {', '.join(missing_core)}")
            print("   请运行: python setup.py --install-deps")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)