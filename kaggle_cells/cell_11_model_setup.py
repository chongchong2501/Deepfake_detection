# Cell 11: 模型初始化和训练配置 - Kaggle T4 GPU优化版本

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler

print("🤖 创建和配置模型...")

# 创建模型 - 针对Kaggle T4 GPU优化
model = OptimizedDeepfakeDetector(
    num_classes=1,
    dropout_rate=0.3,
    use_attention=True,
    use_multimodal=True,  # 启用多模态特征融合
    ensemble_mode=False   # 单模型模式
).to(device)

print(f"✅ 模型已创建并移动到 {device}")
print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 优化GPU内存配置 - 双T4 GPU配置
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # 双T4可以使用更多内存
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# 损失函数 - 使用类别权重平衡
# 计算类别权重 - 修复版本
if hasattr(train_dataset, 'real_count') and hasattr(train_dataset, 'fake_count'):
    # 使用预计算的统计信息
    real_count = train_dataset.real_count
    fake_count = train_dataset.fake_count
else:
    # 回退方案：手动计算
    if hasattr(train_dataset, 'data_list') and train_dataset.data_list is not None:
        real_count = sum(1 for item in train_dataset.data_list if item['label'] == 0)
        fake_count = sum(1 for item in train_dataset.data_list if item['label'] == 1)
    elif hasattr(train_dataset, 'df') and train_dataset.df is not None:
        real_count = len(train_dataset.df[train_dataset.df['label'] == 0])
        fake_count = len(train_dataset.df[train_dataset.df['label'] == 1])
    else:
        # 默认值
        real_count = 1
        fake_count = 1
        print("⚠️ 无法获取类别分布，使用默认权重")

# 确保计数不为零
real_count = max(real_count, 1)
fake_count = max(fake_count, 1)

pos_weight = torch.tensor([real_count / fake_count], device=device)

print(f"📊 类别分布 - 真实: {real_count}, 伪造: {fake_count}")
print(f"⚖️ 正样本权重: {pos_weight.item():.2f}")

# 使用FocalLoss处理类别不平衡
criterion = FocalLoss(
    alpha=0.25,
    gamma=2.0,  # 降低gamma值，减少对困难样本的过度关注
    pos_weight=pos_weight,
    reduction='mean'
)

# 优化器配置 - 使用AdamW和学习率调度
optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-4,  # 提高初始学习率
    weight_decay=1e-4,  # 增加权重衰减
    betas=(0.9, 0.999),
    eps=1e-8
)

# 学习率调度器 - 使用余弦退火
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # 初始重启周期
    T_mult=2,  # 周期倍增因子
    eta_min=1e-6  # 最小学习率
)

# 早停机制 - 双T4 GPU配置
early_stopping = EarlyStopping(
    patience=8,  # 适中的耐心值，适合双T4训练
    min_delta=0.001,
    restore_best_weights=True
)

# 混合精度训练 - 仅在支持的GPU上启用
use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
if use_amp:
    scaler = GradScaler()
    print("✅ 启用混合精度训练 (AMP)")
else:
    scaler = None
    print("📝 使用FP32训练 (兼容性模式)")

# 训练配置 - 双T4 GPU优化
num_epochs = 15  # 适中的训练轮数，适合双T4配置
print(f"🎯 训练配置:")
print(f"  - 训练轮数: {num_epochs}")
print(f"  - 初始学习率: {optimizer.param_groups[0]['lr']:.2e}")
print(f"  - 权重衰减: {optimizer.param_groups[0]['weight_decay']:.2e}")
print(f"  - 早停耐心值: {early_stopping.patience}")
print(f"  - 混合精度: {'启用' if use_amp else '禁用'}")

print("✅ 模型和训练配置完成")