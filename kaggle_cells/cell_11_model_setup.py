# Cell 11: 模型初始化和训练配置 - GPU优化版本

print("🤖 创建和配置高性能模型...")

# 启用GPU优化设置
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("✅ 启用CUDNN优化")

# 创建模型 - 针对T4*2 GPU深度优化
model = OptimizedDeepfakeDetector(
    backbone='resnet50',  # 使用ResNet50以充分利用T4*2 GPU性能
    hidden_dim=768,      # 增加隐藏层维度以充分利用GPU计算能力
    num_layers=3,        # 增加LSTM层数提升模型容量
    dropout=0.4,         # 适当增加dropout防止过拟合
    use_attention=True
).to(device)

# 高效多GPU并行策略 - 充分利用T4*2配置
gpu_count = torch.cuda.device_count()
if gpu_count > 1:
    print(f"🚀 检测到 {gpu_count} 个GPU，启用高性能并行训练")
    print(f"GPU信息: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}")
    
    # 使用DataParallel进行模型并行
    model = nn.DataParallel(model)
    
    # 设置GPU内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%显存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"💾 GPU内存分配: 95% ({total_memory*0.95:.1f}GB per GPU)")
else:
    print("使用单GPU训练")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")
print(f"模型大小估计: {total_params * 4 / 1024**2:.1f} MB")

# 高性能损失函数 - 针对大批次优化
criterion = FocalLoss(alpha=0.25, gamma=2.0)
print(f"损失函数: FocalLoss (alpha=0.25, gamma=2.0) - 大批次优化")

# 高效优化器 - 针对大批次和多GPU优化
base_lr = 0.001
if torch.cuda.device_count() > 1:
    # 多GPU时使用线性缩放学习率
    scaled_lr = base_lr * torch.cuda.device_count() * (batch_size / 8)
    print(f"🔥 多GPU学习率缩放: {base_lr} -> {scaled_lr:.6f}")
else:
    scaled_lr = base_lr * (batch_size / 8)  # 根据批次大小缩放
    print(f"📈 批次大小学习率缩放: {base_lr} -> {scaled_lr:.6f}")

optimizer = optim.AdamW(
    model.parameters(), 
    lr=scaled_lr,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    amsgrad=True  # 启用AMSGrad变体提升稳定性
)
print(f"优化器: AdamW (lr={scaled_lr:.6f}, AMSGrad=True)")

# 高效学习率调度器 - 支持大批次训练
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=scaled_lr * 10,  # 峰值学习率
    epochs=25,  # 使用实际的epoch数
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30%时间用于warm-up
    anneal_strategy='cos',
    div_factor=10,  # 初始学习率 = max_lr / div_factor
    final_div_factor=100  # 最终学习率 = max_lr / final_div_factor
)
print(f"学习率调度器: OneCycleLR (高效大批次训练)")

# 智能早停机制
early_stopping = EarlyStopping(patience=10, min_delta=0.0005)  # 增加patience适应大批次
print(f"早停机制: patience=10, min_delta=0.0005 (大批次优化)")

# 高效混合精度训练
if torch.cuda.is_available():
    scaler = GradScaler(
        init_scale=2.**16,  # 更高的初始缩放因子
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )
    print(f"混合精度训练: 高效配置 (init_scale=65536)")
    print(f"🎯 预期训练加速: 1.5-2x (混合精度 + 大批次 + 多GPU)")
else:
    scaler = None
    print("CPU模式，不使用混合精度训练")

# 训练配置 - 针对T4*2 GPU和更大数据集优化
num_epochs = 25  # 增加训练轮数以充分训练更大的模型
print(f"训练轮数: {num_epochs}")

# 测试模型前向传播
print("\n🔍 测试模型前向传播...")
try:
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        videos, labels = sample_batch
        videos, labels = videos.to(device), labels.to(device)
        
        # 前向传播
        outputs, attention_weights = model(videos)
        loss = criterion(outputs, labels)
        
        print(f"输入形状: {videos.shape}")
        print(f"输出形状: {outputs.shape}")
        print(f"损失值: {loss.item():.4f}")
        print(f"Logits范围: [{outputs.min():.3f}, {outputs.max():.3f}]")
        
        # 显示概率范围
        probs = torch.sigmoid(outputs)
        print(f"概率范围: [{probs.min():.3f}, {probs.max():.3f}]")
        
        if attention_weights is not None:
            print(f"注意力权重形状: {attention_weights.shape}")
        
        print("✅ 模型前向传播测试成功")
except Exception as e:
    print(f"❌ 模型前向传播测试失败: {e}")
    raise e

print("✅ 模型配置完成，准备开始训练")