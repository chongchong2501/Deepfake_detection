# Cell 11: 模型初始化和训练配置 - GPU优化版本

print("🤖 创建和配置模型...")

# 创建模型 - 针对T4*2 GPU优化
model = OptimizedDeepfakeDetector(
    backbone='resnet50',  # 使用ResNet50以充分利用T4*2 GPU性能
    hidden_dim=512,      # 增加隐藏层维度
    num_layers=2,        # 增加LSTM层数
    dropout=0.4,         # 适当增加dropout防止过拟合
    use_attention=True
).to(device)

# 多GPU支持 - 充分利用T4*2配置
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU进行并行训练")
    model = nn.DataParallel(model)
else:
    print("使用单GPU训练")

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")
print(f"模型大小估计: {total_params * 4 / 1024**2:.1f} MB")

# 损失函数
criterion = FocalLoss(alpha=1, gamma=2)
print("使用焦点损失函数 (Focal Loss)")

# 优化器 - 针对ResNet50优化
optimizer = optim.AdamW(
    model.parameters(), 
    lr=2e-4,  # 稍微提高学习率以适应更大模型
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)
print("使用AdamW优化器 (lr=2e-4)")

# 学习率调度器 - 更保守的调度
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.6,  # 更保守的衰减因子
    patience=4,  # 增加patience
    verbose=True,
    min_lr=1e-7
)
print("使用ReduceLROnPlateau学习率调度器 (factor=0.6, patience=4)")

# 早停机制 - 增加patience以适应更大模型
early_stopping = EarlyStopping(patience=8, min_delta=0.001)
print("配置早停机制 (patience=8)")

# 混合精度训练
if torch.cuda.is_available():
    scaler = GradScaler()
    print("启用混合精度训练 (AMP)")
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