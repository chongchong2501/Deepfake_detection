# Cell 11: 模型初始化和训练配置

print("🤖 创建和配置模型...")

# 创建模型
model = OptimizedDeepfakeDetector(
    backbone='resnet18',  # 使用更轻量的backbone以适应Kaggle环境
    hidden_dim=256,
    num_layers=1,
    dropout=0.3,
    use_attention=True
).to(device)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")
print(f"模型大小估计: {total_params * 4 / 1024**2:.1f} MB")

# 损失函数
criterion = FocalLoss(alpha=1, gamma=2)
print("使用焦点损失函数 (Focal Loss)")

# 优化器
optimizer = optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)
print("使用AdamW优化器")

# 学习率调度器
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3, 
    verbose=True,
    min_lr=1e-7
)
print("使用ReduceLROnPlateau学习率调度器")

# 早停机制
early_stopping = EarlyStopping(patience=5, min_delta=0.001)
print("配置早停机制 (patience=5)")

# 混合精度训练
if torch.cuda.is_available():
    scaler = GradScaler()
    print("启用混合精度训练 (AMP)")
else:
    scaler = None
    print("CPU模式，不使用混合精度训练")

# 训练配置
num_epochs = 15  # Kaggle环境下适中的训练轮数
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
        print(f"输出范围: [{outputs.min():.3f}, {outputs.max():.3f}]")
        
        if attention_weights is not None:
            print(f"注意力权重形状: {attention_weights.shape}")
        
        print("✅ 模型前向传播测试成功")
except Exception as e:
    print(f"❌ 模型前向传播测试失败: {e}")
    raise e

print("✅ 模型配置完成，准备开始训练")