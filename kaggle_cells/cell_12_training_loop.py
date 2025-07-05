# Cell 12: 训练循环 - GPU优化版本

print("🚀 开始训练...")

# 训练历史记录
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')
best_model_state = None

# 训练循环
for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*50}")
    
    # 训练阶段 - 使用混合精度
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # 验证阶段 - 使用混合精度
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, scaler)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # 打印结果
    print(f"\n📊 Epoch {epoch+1} 结果:")
    print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
    print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
    print(f"当前学习率: {current_lr:.2e}")
    
    # GPU内存使用情况
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU内存: {memory_allocated:.1f}GB / {memory_reserved:.1f}GB")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        print(f"🎯 新的最佳验证损失: {best_val_loss:.4f}")
    
    # 早停检查
    if early_stopping(val_loss):
        print(f"\n⏹️ 早停触发，在第 {epoch+1} 轮停止训练")
        break
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n✅ 训练完成!")