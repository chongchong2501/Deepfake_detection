# 内存管理优化模块 - RTX4070专用

import torch
import gc
import psutil
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import warnings

@dataclass
class MemoryStats:
    """内存统计信息"""
    cpu_percent: float
    cpu_memory_gb: float
    cpu_memory_percent: float
    gpu_memory_allocated_gb: float
    gpu_memory_cached_gb: float
    gpu_memory_reserved_gb: float
    gpu_memory_percent: float
    timestamp: float

class MemoryManager:
    """智能内存管理器 - RTX4070优化 v2.0"""
    
    def __init__(self, 
                 gpu_memory_threshold: float = 0.75,  # 降低GPU内存使用阈值
                 cpu_memory_threshold: float = 0.85,  # 提高CPU内存使用阈值
                 auto_cleanup_interval: float = 60.0,  # 增加自动清理间隔(秒)
                 enable_monitoring: bool = True,
                 verbose_output: bool = False):  # 新增：控制输出详细程度
        
        self.gpu_memory_threshold = gpu_memory_threshold
        self.cpu_memory_threshold = cpu_memory_threshold
        self.auto_cleanup_interval = auto_cleanup_interval
        self.enable_monitoring = enable_monitoring
        self.verbose_output = verbose_output
        
        # 内存统计历史
        self.memory_history: deque = deque(maxlen=1000)
        
        # 监控线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # 回调函数
        self.cleanup_callbacks: List[Callable] = []
        self.warning_callbacks: List[Callable] = []
        
        # 输出控制
        self.last_cleanup_time = 0
        self.cleanup_message_interval = 30.0  # 清理消息间隔
        
        # GPU信息
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_total_memory = 0
        
        # CPU信息
        self.cpu_total_memory = psutil.virtual_memory().total / (1024**3)
        
        if self.verbose_output:
            print(f"🔧 内存管理器初始化完成")
            print(f"   GPU总内存: {self.gpu_total_memory:.2f}GB")
            print(f"   CPU总内存: {self.cpu_total_memory:.2f}GB")
            print(f"   GPU阈值: {gpu_memory_threshold*100:.0f}%")
            print(f"   CPU阈值: {cpu_memory_threshold*100:.0f}%")
    
    def get_memory_stats(self) -> MemoryStats:
        """获取当前内存统计信息"""
        # CPU统计
        cpu_percent = psutil.cpu_percent()
        cpu_memory = psutil.virtual_memory()
        cpu_memory_gb = cpu_memory.used / (1024**3)
        cpu_memory_percent = cpu_memory.percent / 100.0
        
        # GPU统计
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_memory_percent = gpu_memory_reserved / self.gpu_total_memory
        else:
            gpu_memory_allocated = gpu_memory_cached = gpu_memory_reserved = gpu_memory_percent = 0
        
        return MemoryStats(
            cpu_percent=cpu_percent,
            cpu_memory_gb=cpu_memory_gb,
            cpu_memory_percent=cpu_memory_percent,
            gpu_memory_allocated_gb=gpu_memory_allocated,
            gpu_memory_cached_gb=gpu_memory_cached,
            gpu_memory_reserved_gb=gpu_memory_reserved,
            gpu_memory_percent=gpu_memory_percent,
            timestamp=time.time()
        )
    
    def cleanup_gpu_memory(self, force: bool = False) -> float:
        """清理GPU内存"""
        if not torch.cuda.is_available():
            return 0.0
        
        # 记录清理前的内存
        before_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # 清理缓存
        torch.cuda.empty_cache()
        
        if force:
            # 强制垃圾回收
            gc.collect()
            torch.cuda.empty_cache()
            
            # 同步GPU操作
            torch.cuda.synchronize()
        
        # 记录清理后的内存
        after_memory = torch.cuda.memory_allocated() / (1024**3)
        freed_memory = before_memory - after_memory
        
        if freed_memory > 0.1:  # 释放超过100MB时打印信息
            print(f"🧹 GPU内存清理: 释放了 {freed_memory:.2f}GB")
        
        return freed_memory
    
    def cleanup_cpu_memory(self) -> None:
        """清理CPU内存（减少重复输出）"""
        # 强制垃圾回收
        collected = gc.collect()
        if collected > 0 and self.verbose_output:
            print(f"🧹 CPU内存清理: 回收了 {collected} 个对象")
    
    def smart_cleanup(self) -> Dict[str, float]:
        """智能内存清理（减少重复输出）"""
        stats = self.get_memory_stats()
        results = {'gpu_freed': 0.0, 'cpu_objects': 0}
        current_time = time.time()
        
        # GPU内存清理
        if stats.gpu_memory_percent > self.gpu_memory_threshold:
            # 控制输出频率，避免重复消息
            if (current_time - self.last_cleanup_time) > self.cleanup_message_interval or self.verbose_output:
                print(f"⚠️ GPU内存使用率过高: {stats.gpu_memory_percent*100:.1f}%")
                self.last_cleanup_time = current_time
            
            results['gpu_freed'] = self.cleanup_gpu_memory(force=True)
            
            # 执行注册的清理回调
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    if self.verbose_output:
                        print(f"清理回调执行失败: {e}")
        
        # CPU内存清理（更智能的阈值检查）
        if stats.cpu_memory_percent > self.cpu_memory_threshold:
            # 控制输出频率
            if (current_time - self.last_cleanup_time) > self.cleanup_message_interval or self.verbose_output:
                print(f"⚠️ CPU内存使用率过高: {stats.cpu_memory_percent*100:.1f}%")
                self.last_cleanup_time = current_time
            
            before_objects = len(gc.get_objects())
            self.cleanup_cpu_memory()
            after_objects = len(gc.get_objects())
            results['cpu_objects'] = before_objects - after_objects
        
        return results
    
    def register_cleanup_callback(self, callback: Callable) -> None:
        """注册清理回调函数"""
        self.cleanup_callbacks.append(callback)
    
    def register_warning_callback(self, callback: Callable) -> None:
        """注册警告回调函数"""
        self.warning_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """启动内存监控线程"""
        if not self.enable_monitoring or self.monitoring_thread is not None:
            return
        
        def monitor_loop():
            while not self.stop_monitoring.wait(self.auto_cleanup_interval):
                try:
                    stats = self.get_memory_stats()
                    self.memory_history.append(stats)
                    
                    # 检查是否需要清理
                    if (stats.gpu_memory_percent > self.gpu_memory_threshold or 
                        stats.cpu_memory_percent > self.cpu_memory_threshold):
                        self.smart_cleanup()
                    
                    # 检查是否需要发出警告
                    if stats.gpu_memory_percent > 0.95:
                        for callback in self.warning_callbacks:
                            try:
                                callback(f"GPU内存严重不足: {stats.gpu_memory_percent*100:.1f}%")
                            except Exception as e:
                                print(f"警告回调执行失败: {e}")
                
                except Exception as e:
                    print(f"内存监控错误: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print("📊 内存监控已启动")
    
    def stop_monitoring_thread(self) -> None:
        """停止内存监控线程"""
        if self.monitoring_thread is not None:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
            self.monitoring_thread = None
            print("📊 内存监控已停止")
    
    def print_memory_report(self) -> None:
        """打印详细内存报告"""
        stats = self.get_memory_stats()
        
        print("\n" + "="*50)
        print("📊 内存使用报告")
        print("="*50)
        
        # CPU信息
        print(f"🖥️ CPU:")
        print(f"   使用率: {stats.cpu_percent:.1f}%")
        print(f"   内存: {stats.cpu_memory_gb:.2f}GB / {self.cpu_total_memory:.2f}GB ({stats.cpu_memory_percent*100:.1f}%)")
        
        # GPU信息
        if torch.cuda.is_available():
            print(f"🎮 GPU:")
            print(f"   已分配: {stats.gpu_memory_allocated_gb:.2f}GB")
            print(f"   已缓存: {stats.gpu_memory_cached_gb:.2f}GB")
            print(f"   已保留: {stats.gpu_memory_reserved_gb:.2f}GB / {self.gpu_total_memory:.2f}GB ({stats.gpu_memory_percent*100:.1f}%)")
        
        # 历史统计
        if len(self.memory_history) > 1:
            gpu_usage_history = [s.gpu_memory_percent for s in self.memory_history]
            cpu_usage_history = [s.cpu_memory_percent for s in self.memory_history]
            
            print(f"📈 历史统计 (最近{len(self.memory_history)}次):")
            print(f"   GPU平均使用率: {sum(gpu_usage_history)/len(gpu_usage_history)*100:.1f}%")
            print(f"   GPU峰值使用率: {max(gpu_usage_history)*100:.1f}%")
            print(f"   CPU平均使用率: {sum(cpu_usage_history)/len(cpu_usage_history)*100:.1f}%")
            print(f"   CPU峰值使用率: {max(cpu_usage_history)*100:.1f}%")
        
        print("="*50)
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取内存优化建议"""
        stats = self.get_memory_stats()
        suggestions = []
        
        # GPU优化建议
        if stats.gpu_memory_percent > 0.9:
            suggestions.append("🔴 GPU内存使用率过高，建议减小batch_size")
        elif stats.gpu_memory_percent > 0.8:
            suggestions.append("🟡 GPU内存使用率较高，建议启用梯度累积")
        
        if stats.gpu_memory_cached_gb > stats.gpu_memory_allocated_gb * 1.5:
            suggestions.append("🔵 GPU缓存过多，建议定期调用torch.cuda.empty_cache()")
        
        # CPU优化建议
        if stats.cpu_memory_percent > 0.9:
            suggestions.append("🔴 CPU内存使用率过高，建议减少num_workers")
        elif stats.cpu_memory_percent > 0.8:
            suggestions.append("🟡 CPU内存使用率较高，建议禁用pin_memory")
        
        # 数据加载优化建议
        if len(self.memory_history) > 10:
            recent_gpu_usage = [s.gpu_memory_percent for s in list(self.memory_history)[-10:]]
            if max(recent_gpu_usage) - min(recent_gpu_usage) > 0.3:
                suggestions.append("🔵 GPU内存使用波动较大，建议优化数据预处理")
        
        if not suggestions:
            suggestions.append("✅ 内存使用状况良好")
        
        return suggestions
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop_monitoring_thread()
        self.cleanup_gpu_memory(force=True)
        self.cleanup_cpu_memory()

# 全局内存管理器实例
_global_memory_manager: Optional[MemoryManager] = None

def get_memory_manager() -> MemoryManager:
    """获取全局内存管理器实例"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager

def cleanup_memory(force: bool = False) -> Dict[str, float]:
    """快速内存清理函数"""
    manager = get_memory_manager()
    return manager.smart_cleanup() if not force else {
        'gpu_freed': manager.cleanup_gpu_memory(force=True),
        'cpu_objects': 0
    }

def print_memory_info() -> None:
    """快速打印内存信息"""
    manager = get_memory_manager()
    manager.print_memory_report()

def get_memory_suggestions() -> List[str]:
    """快速获取内存优化建议"""
    manager = get_memory_manager()
    return manager.get_optimization_suggestions()

# 装饰器：自动内存管理
def auto_memory_management(cleanup_interval: int = 50):
    """自动内存管理装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_memory_manager()
            
            # 执行前检查
            if hasattr(wrapper, '_call_count'):
                wrapper._call_count += 1
            else:
                wrapper._call_count = 1
            
            try:
                result = func(*args, **kwargs)
                
                # 定期清理
                if wrapper._call_count % cleanup_interval == 0:
                    manager.smart_cleanup()
                
                return result
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"🚨 检测到内存不足错误，尝试清理内存...")
                    manager.cleanup_gpu_memory(force=True)
                    manager.cleanup_cpu_memory()
                    print(f"💡 建议: {'; '.join(manager.get_optimization_suggestions())}")
                raise
        
        return wrapper
    return decorator