# Cell 12: 模型训练主循环

print("🎯 开始训练模型...")
print("=" * 60)

# 训练历史记录
train_history = {
    'train_loss': [],
    'train_acc': [],
    'train_auc': [],
    'val_loss': [],
    'val_acc': [],
    'val_auc': [],
    'lr': []
}

best_val_acc = 0
best_val_auc = 0
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    # 训练阶段
    train_loss, train_acc, train_auc = train_epoch(
        model, train_loader, criterion, optimizer, device, scaler
    )
    
    # 验证阶段
    val_loss, val_acc, val_auc = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录历史
    train_history['train_loss'].append(train_loss)
    train_history['train_acc'].append(train_acc)
    train_history['train_auc'].append(train_auc)
    train_history['val_loss'].append(val_loss)
    train_history['val_acc'].append(val_acc)
    train_history['val_auc'].append(val_auc)
    train_history['lr'].append(current_lr)
    
    # 计算epoch时间
    epoch_time = time.time() - epoch_start_time
    
    # 打印结果
    print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%, AUC: {train_auc:.4f}")
    print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%, AUC: {val_auc:.4f}")
    print(f"学习率: {current_lr:.6f}")
    print(f"Epoch时间: {epoch_time:.1f}s")
    
    # 保存最佳模型
    is_best = False
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_auc = val_auc
        is_best = True
        
        # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_val_auc': best_val_auc,
            'val_loss': val_loss,
            'train_history': train_history
        }, './models/best_model.pth')
        
        print(f"💾 保存最佳模型 (验证准确率: {val_acc:.2f}%, AUC: {val_auc:.4f})")
    
    # 早停检查
    if early_stopping(val_loss, model):
        print(f"🛑 早停触发，在第 {epoch+1} 轮停止训练")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        break
    
    # 内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# 训练完成
total_time = time.time() - start_time
print(f"\n🎉 训练完成！")
print(f"总训练时间: {total_time/60:.1f} 分钟")
print(f"最佳验证准确率: {best_val_acc:.2f}%")
print(f"最佳验证AUC: {best_val_auc:.4f}")
print("=" * 60)

# 绘制训练历史
print("📊 绘制训练历史...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 损失曲线
ax1.plot(train_history['train_loss'], label='训练损失', color='blue')
ax1.plot(train_history['val_loss'], label='验证损失', color='red')
ax1.set_title('损失曲线')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 准确率曲线
ax2.plot(train_history['train_acc'], label='训练准确率', color='blue')
ax2.plot(train_history['val_acc'], label='验证准确率', color='red')
ax2.set_title('准确率曲线')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# AUC曲线
ax3.plot(train_history['train_auc'], label='训练AUC', color='blue')
ax3.plot(train_history['val_auc'], label='验证AUC', color='red')
ax3.set_title('AUC曲线')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('AUC')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 学习率曲线
ax4.plot(train_history['lr'], label='学习率', color='green')
ax4.set_title('学习率变化')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./results/training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 训练历史图表已保存")
print("✅ 训练阶段完成，准备进行模型评估")