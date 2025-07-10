# Cell 11: 模型初始化和训练配置 - Kaggle T4 GPU优化版本

print("🤖 创建和配置模型...")

# 创建模型 - 针对Kaggle T4 GPU优化
model = OptimizedDeepfakeDetector(
    backbone='resnet50',
    hidden_dim=512,      # 适中的隐藏层维度
    num_layers=2,        # 减少LSTM层数
    dropout=0.3,         # 适中的dropout
    use_attention=True
).to(device)

# 单GPU配置
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)
    print("使用单GPU训练")

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")
print(f"模型大小估计: {total_params * 4 / 1024**2:.1f} MB")

# 损失函数 - 针对严重类别不平衡优化
# 计算类别权重（假设真实视频是少数类）
pos_weight = torch.tensor([3.0]).to(device)  # 给真实视频更高权重
criterion = FocalLoss(alpha=0.75, gamma=3.0, pos_weight=pos_weight)  # 增强对困难样本的关注
print(f"损失函数: FocalLoss (alpha=0.75, gamma=3.0, pos_weight=3.0)")

# 优化器
base_lr = 0.001
optimizer = optim.AdamW(
    model.parameters(), 
    lr=base_lr,
    weight_decay=0.01
)
print(f"优化器: AdamW (lr={base_lr})")

# 学习率调度器
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=base_lr * 5,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)
print(f"学习率调度器: OneCycleLR")

# 早停机制
early_stopping = EarlyStopping(patience=7, min_delta=0.001)
print(f"早停机制: patience=7, min_delta=0.001")

# 训练配置 - 统一使用FP32数据类型
scaler = None
print("数据类型: FP32 (确保兼容性)")

num_epochs = 20
print(f"训练轮数: {num_epochs}")

# 测试模型前向传播
print("\n🔍 测试模型前向传播...")
try:
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        videos, labels = sample_batch
        videos, labels = videos.to(device), labels.to(device)
        
        # 前向传播（统一使用FP32）
        outputs, attention_weights = model(videos)
        loss = criterion(outputs, labels)
        
        print(f"输入形状: {videos.shape}")
        print(f"输入数据类型: {videos.dtype}")
        print(f"输出形状: {outputs.shape}")
        print(f"损失值: {loss.item():.4f}")
        
        # 显示概率范围
        probs = torch.sigmoid(outputs)
        print(f"概率范围: [{probs.min():.3f}, {probs.max():.3f}]")
        
        print("✅ 模型前向传播测试成功")
except Exception as e:
    print(f"❌ 模型前向传播测试失败: {e}")
    raise e

print("✅ 模型配置完成，准备开始训练")