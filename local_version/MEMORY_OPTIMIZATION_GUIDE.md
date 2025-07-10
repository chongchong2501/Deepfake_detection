# 内存优化指南 - RTX4070深度伪造检测

## 概述

本指南介绍了为RTX4070显卡（12GB显存）专门设计的内存管理和优化系统。通过智能内存监控、自动清理和参数调优，显著提升训练稳定性和效率。

## 🚀 快速开始

### 使用内存优化配置

```bash
# 使用专门的内存优化配置
cd local_version
python train.py --config configs/memory_optimized.yaml
```

### 手动内存管理

```python
from src.memory_manager import MemoryManager, print_memory_info

# 创建内存管理器
with MemoryManager() as manager:
    # 你的训练代码
    trainer.train(epochs=20)
```

## 📊 内存管理功能

### 1. 智能内存监控

- **实时监控**: 持续监控CPU和GPU内存使用情况
- **自动阈值检测**: 当内存使用超过设定阈值时自动触发清理
- **历史统计**: 记录内存使用历史，提供性能分析

### 2. 自动内存清理

- **GPU缓存清理**: 自动清理PyTorch GPU缓存
- **CPU垃圾回收**: 强制Python垃圾回收
- **数据集缓存清理**: 清理视频帧缓存
- **模型缓存清理**: 清理模型中间结果缓存

### 3. 智能参数调优

- **动态批次大小**: 根据内存使用情况自动调整batch_size
- **工作进程优化**: 自动调整num_workers数量
- **预处理模式切换**: 在GPU/CPU预处理间智能切换

## ⚙️ 配置选项

### 内存管理器配置

```python
MemoryManager(
    gpu_memory_threshold=0.85,    # GPU内存使用阈值（85%）
    cpu_memory_threshold=0.80,    # CPU内存使用阈值（80%）
    auto_cleanup_interval=30.0,   # 自动清理间隔（秒）
    enable_monitoring=True        # 启用实时监控
)
```

### YAML配置文件

```yaml
gpu:
  memory_management:
    gpu_threshold: 0.75        # GPU内存阈值
    cpu_threshold: 0.75        # CPU内存阈值
    auto_cleanup_interval: 25  # 自动清理间隔
    enable_monitoring: true    # 启用内存监控
    force_cleanup_every: 100   # 强制清理间隔
```

## 🎯 优化策略

### 1. 显存优化（12GB RTX4070）

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| batch_size | 6-8 | 根据模型复杂度调整 |
| max_frames | 16-24 | 平衡性能和内存使用 |
| backbone | resnet18/resnet50 | 避免过大的模型 |
| mixed_precision | true | 启用FP16混合精度 |
| pin_memory | false | 避免CUDA tensor错误 |

### 2. 系统内存优化

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| num_workers | 2-4 | 避免过多进程 |
| enable_cache | false | 禁用缓存节省内存 |
| prefetch_factor | 1-2 | 减少预取数据量 |
| persistent_workers | true | 重用工作进程 |

### 3. 数据处理优化

- **GPU预处理**: 仅在显存充足时启用
- **帧缓存**: 小数据集启用，大数据集禁用
- **质量过滤**: 适当降低阈值保留更多帧
- **数据增强**: 使用轻量级增强策略

## 🔧 故障排除

### 常见内存错误及解决方案

#### 1. CUDA Out of Memory

**错误信息**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 自动处理（已集成）
@auto_memory_management(cleanup_interval=50)
def train_step(self, batch):
    # 训练代码
    pass

# 手动处理
try:
    output = model(input)
except RuntimeError as e:
    if "out of memory" in str(e):
        cleanup_memory(force=True)
        # 重试或降低batch_size
```

#### 2. Pin Memory错误

**错误信息**: `cannot pin 'torch.cuda.FloatTensor'`

**解决方案**:
```yaml
dataloader:
  pin_memory: false  # 在配置文件中禁用
```

#### 3. 工作进程内存泄漏

**解决方案**:
```yaml
dataloader:
  num_workers: 2           # 减少工作进程
  persistent_workers: true # 重用进程
```

### 内存使用监控

```python
# 查看当前内存状态
from src.memory_manager import print_memory_info
print_memory_info()

# 获取优化建议
from src.memory_manager import get_memory_suggestions
suggestions = get_memory_suggestions()
for suggestion in suggestions:
    print(suggestion)
```

## 📈 性能基准

### RTX4070 (12GB) 性能表现

| 配置 | Batch Size | 显存使用 | 训练速度 | 稳定性 |
|------|------------|----------|----------|--------|
| default.yaml | 12 | ~11GB | 快 | 不稳定 |
| medium.yaml | 10 | ~9GB | 中等 | 较稳定 |
| memory_optimized.yaml | 6 | ~7GB | 稍慢 | 非常稳定 |

### 内存使用模式

- **训练阶段**: 显存使用75-85%
- **验证阶段**: 显存使用60-70%
- **峰值内存**: 通常在epoch开始时
- **清理效果**: 每次清理可释放1-2GB显存

## 🛠️ 高级功能

### 1. 自定义内存回调

```python
def custom_cleanup():
    # 自定义清理逻辑
    torch.cuda.empty_cache()
    gc.collect()

# 注册回调
memory_manager.register_cleanup_callback(custom_cleanup)
```

### 2. 内存分析和调试

```python
# 启用内存分析
with MemoryManager(enable_monitoring=True) as manager:
    # 训练代码
    history = trainer.train(epochs=10)
    
    # 获取内存统计
    stats = manager.get_memory_stats()
    print(f"峰值GPU内存: {stats.gpu_memory_reserved_gb:.2f}GB")
```

### 3. 梯度累积优化

```python
# 在内存不足时使用梯度累积
accumulation_steps = 2
for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📋 最佳实践

### 1. 训练前检查

```bash
# 检查系统资源
nvidia-smi
free -h

# 运行内存测试
python -c "from src.memory_manager import print_memory_info; print_memory_info()"
```

### 2. 配置选择指南

- **首次训练**: 使用 `memory_optimized.yaml`
- **调试阶段**: 使用 `quick_demo.yaml`
- **生产训练**: 根据硬件选择 `medium.yaml` 或 `default.yaml`

### 3. 监控和调优

- 定期查看内存报告
- 根据建议调整参数
- 记录最佳配置组合

## 🔍 常见问题

**Q: 为什么要禁用pin_memory？**
A: 在某些PyTorch版本中，pin_memory可能导致CUDA tensor错误。禁用后可提高稳定性，对性能影响很小。

**Q: 内存清理会影响训练精度吗？**
A: 不会。内存清理只清除缓存和临时数据，不影响模型参数和梯度。

**Q: 如何选择合适的batch_size？**
A: 从小开始（如4），逐步增加直到显存使用率达到80-85%。

**Q: 混合精度训练安全吗？**
A: 是的。现代GPU都支持FP16，可以显著节省显存而不影响精度。

## 📞 技术支持

如果遇到内存相关问题：

1. 查看内存报告和优化建议
2. 尝试使用 `memory_optimized.yaml` 配置
3. 检查系统资源使用情况
4. 参考故障排除部分

---

*本指南针对RTX4070显卡优化，其他显卡可能需要调整参数。*