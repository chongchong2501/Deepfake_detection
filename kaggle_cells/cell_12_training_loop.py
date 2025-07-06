# Cell 12: 训练循环 - GPU优化版本
# 所有import语句已移至cell_01_imports_and_setup.py

# 确保模型保存目录存在
os.makedirs('./models', exist_ok=True)

print("🚀 开始训练...")
print(f"📊 训练配置: {len(train_loader)} 个训练批次, {len(val_loader)} 个验证批次")
print(f"🎯 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"💾 设备: {device}")
print(f"🔧 数据加载优化: {num_workers} workers, prefetch={prefetch_factor}, persistent={persistent_workers}")
print(f"📦 批次大小: {batch_size}, 帧缓存: {'启用' if train_dataset.cache_frames else '禁用'}")

# 性能监控
if torch.cuda.is_available():
    print(f"🎮 GPU信息: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.reset_peak_memory_stats()

# 训练历史记录
train_history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'train_auc': [],
    'val_auc': [],
    'lr': [],
    'gpu_memory': [],
    'epoch_time': []
}
best_val_loss = float('inf')
best_val_acc = 0.0
best_val_auc = 0.0
best_model_state = None

# 训练循环
print("\n🔄 开始训练循环...")
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs} - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # 训练阶段 - 使用混合精度
    print(f"📚 开始训练第 {epoch+1} 轮...")
    train_start = time.time()
    train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
    train_time = time.time() - train_start
    
    # 验证阶段 - 使用混合精度
    print(f"🔍 开始验证第 {epoch+1} 轮...")
    val_start = time.time()
    val_loss, val_acc, val_auc = validate_epoch(model, val_loader, criterion, device, scaler)
    val_time = time.time() - val_start
    
    # 记录历史
    train_history['train_loss'].append(train_loss)
    train_history['train_acc'].append(train_acc)
    train_history['train_auc'].append(train_auc)
    train_history['val_loss'].append(val_loss)
    train_history['val_acc'].append(val_acc)
    train_history['val_auc'].append(val_auc)
    
    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    train_history['lr'].append(current_lr)
    
    # 计算epoch时间
    epoch_time = time.time() - epoch_start_time
    train_history['epoch_time'].append(epoch_time)
    
    # GPU内存使用情况
    gpu_memory = 0
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_peak = torch.cuda.max_memory_allocated() / 1024**3
        gpu_memory = memory_allocated
        train_history['gpu_memory'].append(gpu_memory)
    
    # 打印详细结果
    print(f"\n📊 Epoch {epoch+1} 结果:")
    print(f"训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, AUC={train_auc:.4f} (用时: {train_time:.1f}s)")
    print(f"验证: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc:.4f} (用时: {val_time:.1f}s)")
    print(f"学习率: {current_lr:.2e}, Epoch用时: {epoch_time:.1f}s")
    
    if torch.cuda.is_available():
        print(f"GPU内存: {memory_allocated:.1f}GB/{memory_reserved:.1f}GB (峰值: {memory_peak:.1f}GB)")
    
    # 保存最佳模型（基于多个指标）
    is_best = False
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_val_auc = val_auc
        best_model_state = model.state_dict().copy()
        is_best = True
        print(f"🎯 新的最佳模型! Loss: {best_val_loss:.4f}, Acc: {best_val_acc:.2f}%, AUC: {best_val_auc:.4f}")
        
        # 保存最佳模型到文件
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_val_auc': best_val_auc,
            'train_history': train_history
        }, './models/best_model.pth')
        print("💾 最佳模型已保存到 ./models/best_model.pth")
    
    # 早停检查
    if early_stopping(val_loss, model):
        print(f"\n⏹️ 早停触发，在第 {epoch+1} 轮停止训练")
        print(f"最佳性能: Loss={best_val_loss:.4f}, Acc={best_val_acc:.2f}%, AUC={best_val_auc:.4f}")
        break
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n✅ 训练完成!")
print(f"🏆 最终最佳性能: Loss={best_val_loss:.4f}, Acc={best_val_acc:.2f}%, AUC={best_val_auc:.4f}")
print(f"⏱️ 总训练时间: {sum(train_history['epoch_time']):.1f}秒")
if torch.cuda.is_available():
    print(f"💾 峰值GPU内存使用: {torch.cuda.max_memory_allocated() / 1024**3:.1f}GB")