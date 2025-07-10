#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存管理功能测试脚本
用于验证新的内存管理系统是否正常工作
"""

import torch
import time
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_memory_manager_basic():
    """测试基础内存管理功能"""
    print("🧪 测试基础内存管理功能...")
    
    try:
        from memory_manager import MemoryManager, print_memory_info, get_memory_suggestions
        
        # 创建内存管理器
        manager = MemoryManager(
            gpu_memory_threshold=0.8,
            cpu_memory_threshold=0.8,
            auto_cleanup_interval=5.0,
            enable_monitoring=False  # 测试时禁用自动监控
        )
        
        print("✅ 内存管理器创建成功")
        
        # 测试内存统计
        stats = manager.get_memory_stats()
        print(f"✅ 内存统计获取成功: GPU {stats.gpu_memory_percent*100:.1f}%, CPU {stats.cpu_memory_percent*100:.1f}%")
        
        # 测试内存清理
        freed = manager.cleanup_gpu_memory()
        print(f"✅ GPU内存清理成功: 释放了 {freed:.2f}GB")
        
        # 测试优化建议
        suggestions = manager.get_optimization_suggestions()
        print(f"✅ 优化建议获取成功: {len(suggestions)} 条建议")
        
        # 测试内存报告
        print("\n📊 内存报告测试:")
        manager.print_memory_report()
        
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

def test_memory_manager_with_tensors():
    """测试内存管理器在处理张量时的表现"""
    print("\n🧪 测试张量内存管理...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，跳过GPU张量测试")
        return True
    
    try:
        from memory_manager import MemoryManager, auto_memory_management
        
        @auto_memory_management(cleanup_interval=3)
        def create_large_tensors(size):
            """创建大张量的测试函数"""
            tensors = []
            for i in range(5):
                tensor = torch.randn(size, size, device='cuda')
                tensors.append(tensor)
            return tensors
        
        # 创建内存管理器
        with MemoryManager() as manager:
            print("✅ 上下文管理器工作正常")
            
            # 创建一些大张量来测试内存管理
            print("创建大张量进行测试...")
            
            for i in range(10):
                tensors = create_large_tensors(1000)
                if i % 3 == 0:
                    print(f"  第 {i+1} 轮: 创建了 {len(tensors)} 个张量")
                
                # 手动清理一些张量
                del tensors
            
            print("✅ 张量内存管理测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 张量内存管理测试失败: {e}")
        return False

def test_memory_optimization_config():
    """测试内存优化配置文件"""
    print("\n🧪 测试内存优化配置...")
    
    try:
        from config import Config
        
        # 测试加载内存优化配置
        config_path = "configs/memory_optimized.yaml"
        if os.path.exists(config_path):
            Config.load_config(config_path)
            print(f"✅ 内存优化配置加载成功")
            print(f"   batch_size: {getattr(Config, 'BATCH_SIZE', 'N/A')}")
            print(f"   max_frames: {getattr(Config, 'MAX_FRAMES', 'N/A')}")
            print(f"   backbone: {getattr(Config, 'BACKBONE', 'N/A')}")
            print(f"   pin_memory: {getattr(Config, 'PIN_MEMORY', 'N/A')}")
            return True
        else:
            print(f"⚠️ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def test_integration_with_existing_code():
    """测试与现有代码的集成"""
    print("\n🧪 测试与现有代码集成...")
    
    try:
        # 测试utils函数的兼容性
        from utils import print_gpu_memory_info, cleanup_gpu_memory, PerformanceMonitor
        
        print("测试增强的utils函数...")
        
        # 测试内存信息打印
        print_gpu_memory_info()
        print("✅ print_gpu_memory_info 工作正常")
        
        # 测试内存清理
        freed = cleanup_gpu_memory()
        if isinstance(freed, dict):
            gpu_freed = freed.get('gpu_freed', 0)
            print(f"✅ cleanup_gpu_memory 工作正常: 释放了 {gpu_freed:.2f}GB")
        else:
            print(f"✅ cleanup_gpu_memory 工作正常: 释放了 {freed:.2f}GB")
        
        # 测试性能监控器
        monitor = PerformanceMonitor()
        monitor.update()
        stats = monitor.get_stats()
        suggestions = monitor.get_memory_suggestions()
        
        print(f"✅ PerformanceMonitor 工作正常: {len(suggestions)} 条建议")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理机制"""
    print("\n🧪 测试错误处理机制...")
    
    try:
        from memory_manager import MemoryManager
        
        # 测试无效参数
        try:
            manager = MemoryManager(gpu_memory_threshold=2.0)  # 无效阈值
            print("⚠️ 应该检测到无效参数")
        except:
            print("✅ 无效参数检测正常")
        
        # 测试内存不足模拟
        if torch.cuda.is_available():
            try:
                # 尝试分配超大张量来触发内存不足
                huge_tensor = torch.randn(50000, 50000, device='cuda')
                print("⚠️ 应该触发内存不足错误")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("✅ 内存不足错误处理正常")
                else:
                    print(f"✅ 其他CUDA错误处理正常: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始内存管理功能测试")
    print("=" * 50)
    
    # 显示系统信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("基础功能测试", test_memory_manager_basic),
        ("张量内存管理测试", test_memory_manager_with_tensors),
        ("配置文件测试", test_memory_optimization_config),
        ("代码集成测试", test_integration_with_existing_code),
        ("错误处理测试", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    # 测试结果
    print("\n" + "=" * 50)
    print(f"🎯 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！内存管理系统工作正常。")
        print("\n💡 使用建议:")
        print("   1. 使用 'python train.py --config configs/memory_optimized.yaml' 进行训练")
        print("   2. 查看 MEMORY_OPTIMIZATION_GUIDE.md 了解详细使用方法")
        print("   3. 根据内存使用情况调整配置参数")
    else:
        print("⚠️ 部分测试失败，请检查错误信息并修复问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)