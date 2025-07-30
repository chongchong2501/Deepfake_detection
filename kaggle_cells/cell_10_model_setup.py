# Cell 10: 模型初始化和训练配置 
print("🤖 创建和配置模型...")

# 训练配置参数 - ResNet18优化版本
batch_size = 16  # 增加批次大小，ResNet18内存占用更小，可以使用更大批次

# 创建轻量化模型 - ResNet18版本
model = OptimizedDeepfakeDetector(
    num_classes=1,
    dropout_rate=0.2,  # 适中的dropout率，平衡过拟合和欠拟合
    use_attention=False,  # 禁用注意力机制，简化模型
    use_multimodal=False,  # 禁用多模态特征融合
    ensemble_mode=False   # 单模型模式
).to(device)

# 多GPU并行支持
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"🚀 启用多GPU并行训练，使用 {torch.cuda.device_count()} 个GPU")
    model = nn.DataParallel(model)
    # 调整批次大小以充分利用多GPU
    effective_batch_size = batch_size * torch.cuda.device_count()
    print(f"📦 有效批次大小: {effective_batch_size} (单GPU: {batch_size})")
else:
    print("📝 单GPU训练模式")

print(f"✅ 模型已创建并移动到 {device}")
print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 优化GPU内存配置 - 更保守的内存使用避免OOM
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.6)  # 降低到60%避免内存溢出
    torch.cuda.empty_cache()  # 清理缓存
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"🔧 内存使用限制: 60%")

# 损失函数 - 使用类别权重平衡
# 计算类别权重 - 修复版本
if 'train_loader' in globals() and train_loader is not None:
    # 从train_loader获取数据集
    train_dataset = train_loader.dataset
    
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
else:
    # 如果没有train_loader，使用默认值
    real_count = 1
    fake_count = 1
    print("⚠️ train_loader未定义，使用默认类别权重")

# 确保计数不为零
real_count = max(real_count, 1)
fake_count = max(fake_count, 1)

pos_weight = torch.tensor([real_count / fake_count], device=device)

print(f"📊 类别分布 - 真实: {real_count}, 伪造: {fake_count}")
print(f"⚖️ 正样本权重: {pos_weight.item():.2f}")

# 使用FocalLoss处理类别不平衡 - ResNet18优化版本
criterion = FocalLoss(
    alpha=0.9,   # 增加alpha值，更多关注真实视频(少数类)
    gamma=2.0,   # 适中的gamma值，关注困难样本但不过度
    pos_weight=pos_weight,
    reduction='mean'
)

# 优化器配置 - ResNet18优化版本
optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-4,  # 适中的学习率，适合ResNet18的收敛特性
    weight_decay=0.01,  # 适中的权重衰减，防止过拟合
    betas=(0.9, 0.999),
    eps=1e-8
)

# 学习率调度器 - 更激进的学习率策略
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,  # 增加重启周期，让模型有更多时间学习
    T_mult=1,  # 保持周期不变
    eta_min=1e-5  # 提高最小学习率
)

# 早停机制 - 更宽松的监控
early_stopping = EarlyStopping(
    patience=10,  # 增加耐心值，给模型更多学习时间
    min_delta=0.005,  # 降低最小改进阈值
    restore_best_weights=True
)

# 混合精度训练 - 暂时禁用以解决NaN问题
use_amp = False  # 强制禁用混合精度训练，避免数值不稳定
scaler = None
print("📝 使用FP32训练 (解决NaN问题)")

# 训练配置 - ResNet18优化版本
num_epochs = 50  # 保持训练轮数，给模型充分学习时间
print(f"🎯 ResNet18优化训练配置:")
print(f"  - 主干网络: ResNet18 (轻量化)")
print(f"  - 批次大小: {batch_size} (相比ResNet50增加100%)")
print(f"  - 训练轮数: {num_epochs}")
print(f"  - 初始学习率: {optimizer.param_groups[0]['lr']:.2e}")
print(f"  - 权重衰减: {optimizer.param_groups[0]['weight_decay']:.3f}")
print(f"  - FocalLoss参数: alpha={criterion.alpha:.1f}, gamma={criterion.gamma:.1f}")
print(f"  - 早停耐心值: {early_stopping.patience}")
print(f"  - 混合精度: {'启用' if use_amp else '禁用'}")
print(f"  - 模型复杂度: 轻量化版本 (~11M参数)")

print("✅ ResNet18轻量化模型和训练配置完成")