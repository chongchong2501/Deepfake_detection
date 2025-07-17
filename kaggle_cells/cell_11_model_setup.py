# Cell 11: 模型初始化和训练配置 - Kaggle T4 GPU优化版本

import torch.nn as nn

print("🤖 创建和配置模型...")

# 创建模型 - 使用更强的EfficientNet backbone
model = OptimizedDeepfakeDetector(
    backbone='efficientnet_b0',  # 使用EfficientNet
    hidden_dim=512,
    num_layers=2,
    dropout=0.3,
    use_attention=True
)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"使用多GPU训练: {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.to(device)

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

# 损失函数 - 使用平衡的配置，移除pos_weight偏向
criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=None)  # 更平衡的参数
print(f"损失函数: FocalLoss (alpha=0.25, gamma=2.0, 无pos_weight偏向)")

# 优化器 - 降低学习率
base_lr = 0.0001  # 降低学习率
optimizer = optim.AdamW(
    model.parameters(), 
    lr=base_lr,
    weight_decay=0.01
)
print(f"优化器: AdamW (lr={base_lr})")

# 学习率调度器 - 增加训练轮数
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=base_lr * 10,  # 调整最大学习率
    epochs=50,  # 增加训练轮数
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)
print(f"学习率调度器: OneCycleLR (50 epochs)")

# 早停机制 - 增加patience
early_stopping = EarlyStopping(patience=15, min_delta=0.001)  # 增加patience
print(f"早停机制: patience=15, min_delta=0.001")

# 训练配置 - 统一使用FP32数据类型
scaler = None
print("数据类型: FP32 (确保兼容性)")

num_epochs = 50  # 增加训练轮数
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