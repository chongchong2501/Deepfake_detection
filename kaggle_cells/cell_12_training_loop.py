# Cell 12: 训练循环 - 集成优化版本

# 确保模型保存目录存在
import os
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

os.makedirs('./models', exist_ok=True)

# 检查是否启用集成学习
enable_ensemble = getattr(model, 'ensemble_mode', False)
if enable_ensemble:
    print("🎯 启用集成学习模式")
    # 创建多个模型用于集成
    from .cell_05_model_definition import create_ensemble_models
    ensemble_models = create_ensemble_models(num_models=3, device=device)
    print(f"📊 创建了 {len(ensemble_models)} 个集成模型")
else:
    ensemble_models = [model]

print("🚀 开始训练...")
print(f"📊 训练配置: {len(train_loader)} 个训练批次, {len(val_loader)} 个验证批次")
print(f"🎯 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"💾 设备: {device}")
print(f"📦 批次大小: {batch_size}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

# 训练历史记录 - 扩展版本
train_history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'train_auc': [],
    'val_auc': [],
    'train_precision': [],
    'val_precision': [],
    'train_recall': [],
    'val_recall': [],
    'train_f1': [],
    'val_f1': []
}

# 集成学习历史记录
if enable_ensemble:
    ensemble_history = {
        'ensemble_val_acc': [],
        'ensemble_val_auc': [],
        'ensemble_val_f1': [],
        'individual_performances': []
    }

best_val_loss = float('inf')
best_val_acc = 0.0
best_val_auc = 0.0
best_ensemble_acc = 0.0
best_model_states = []

# 训练循环
print("\n🔄 开始训练循环...")
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    if enable_ensemble:
        # 集成训练
        from .cell_07_training_functions import train_ensemble_models, ensemble_predict
        
        # 训练所有集成模型
        ensemble_train_results = train_ensemble_models(
            ensemble_models, train_loader, criterion, 
            [optimizer] * len(ensemble_models), device, scaler
        )
        
        # 计算平均训练指标
        train_loss = np.mean([r['loss'] for r in ensemble_train_results])
        train_acc = np.mean([r['accuracy'] for r in ensemble_train_results])
        train_auc = np.mean([r['auc'] for r in ensemble_train_results])
        train_precision = np.mean([r['precision'] for r in ensemble_train_results])
        train_recall = np.mean([r['recall'] for r in ensemble_train_results])
        train_f1 = np.mean([r['f1'] for r in ensemble_train_results])
        
        # 集成验证
        val_results = []
        all_val_preds = []
        all_val_labels = []
        
        for model_idx, ens_model in enumerate(ensemble_models):
            val_loss_single, val_acc_single, val_auc_single, val_metrics = validate_epoch(
                ens_model, val_loader, criterion, device, scaler, return_detailed_metrics=True
            )
            val_results.append({
                'loss': val_loss_single,
                'accuracy': val_acc_single,
                'auc': val_auc_single,
                **val_metrics
            })
        
        # 计算集成预测结果
        ensemble_val_acc, ensemble_val_auc, ensemble_val_f1 = ensemble_predict(
            ensemble_models, val_loader, device
        )
        
        # 计算平均验证指标
        val_loss = np.mean([r['loss'] for r in val_results])
        val_acc = np.mean([r['accuracy'] for r in val_results])
        val_auc = np.mean([r['auc'] for r in val_results])
        val_precision = np.mean([r['precision'] for r in val_results])
        val_recall = np.mean([r['recall'] for r in val_results])
        val_f1 = np.mean([r['f1'] for r in val_results])
        
        # 记录集成历史
        ensemble_history['ensemble_val_acc'].append(ensemble_val_acc)
        ensemble_history['ensemble_val_auc'].append(ensemble_val_auc)
        ensemble_history['ensemble_val_f1'].append(ensemble_val_f1)
        ensemble_history['individual_performances'].append(val_results)
        
    else:
        # 单模型训练
        train_loss, train_acc, train_auc, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, return_detailed_metrics=True
        )
        val_loss, val_acc, val_auc, val_metrics = validate_epoch(
            model, val_loader, criterion, device, scaler, return_detailed_metrics=True
        )
        
        train_precision = train_metrics['precision']
        train_recall = train_metrics['recall']
        train_f1 = train_metrics['f1']
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_f1 = val_metrics['f1']
    
    # 记录历史
    train_history['train_loss'].append(train_loss)
    train_history['train_acc'].append(train_acc)
    train_history['train_auc'].append(train_auc)
    train_history['train_precision'].append(train_precision)
    train_history['train_recall'].append(train_recall)
    train_history['train_f1'].append(train_f1)
    train_history['val_loss'].append(val_loss)
    train_history['val_acc'].append(val_acc)
    train_history['val_auc'].append(val_auc)
    train_history['val_precision'].append(val_precision)
    train_history['val_recall'].append(val_recall)
    train_history['val_f1'].append(val_f1)
    
    # 学习率调度
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # 计算epoch时间
    epoch_time = time.time() - epoch_start_time
    
    # 打印结果
    print(f"训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, AUC={train_auc:.4f}, F1={train_f1:.4f}")
    print(f"验证: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, AUC={val_auc:.4f}, F1={val_f1:.4f}")
    
    if enable_ensemble:
        print(f"集成: Acc={ensemble_val_acc:.2f}%, AUC={ensemble_val_auc:.4f}, F1={ensemble_val_f1:.4f}")
    
    print(f"学习率: {current_lr:.2e}, 用时: {epoch_time:.1f}s")
    
    # 保存最佳模型
    current_metric = ensemble_val_acc if enable_ensemble else val_acc
    if current_metric > best_val_acc:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_val_auc = val_auc
        
        if enable_ensemble:
            best_ensemble_acc = ensemble_val_acc
            best_model_states = [model.state_dict().copy() for model in ensemble_models]
            print(f"🎯 新的最佳集成模型! 集成Acc: {best_ensemble_acc:.2f}%, 平均AUC: {best_val_auc:.4f}")
        else:
            best_model_states = [model.state_dict().copy()]
            print(f"🎯 新的最佳模型! Acc: {best_val_acc:.2f}%, AUC: {best_val_auc:.4f}")
        
        # 保存最佳模型到文件
        save_dict = {
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_val_auc': best_val_auc,
            'train_history': train_history,
            'enable_ensemble': enable_ensemble,
            'num_models': len(best_model_states)
        }
        
        if enable_ensemble:
            save_dict['best_ensemble_acc'] = best_ensemble_acc
            save_dict['ensemble_history'] = ensemble_history
            for i, state in enumerate(best_model_states):
                save_dict[f'model_{i}_state_dict'] = state
        else:
            save_dict['model_state_dict'] = best_model_states[0]
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, './models/best_model.pth')
        print("💾 最佳模型已保存")
    
    # 早停检查
    if early_stopping(val_loss, model):
        print(f"\n⏹️ 早停触发，在第 {epoch+1} 轮停止训练")
        break
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n✅ 训练完成!")
if enable_ensemble:
    print(f"🏆 最终最佳性能: 集成Acc={best_ensemble_acc:.2f}%, 平均Loss={best_val_loss:.4f}, 平均AUC={best_val_auc:.4f}")
else:
    print(f"🏆 最终最佳性能: Loss={best_val_loss:.4f}, Acc={best_val_acc:.2f}%, AUC={best_val_auc:.4f}")

if torch.cuda.is_available():
    print(f"💾 峰值GPU内存使用: {torch.cuda.max_memory_allocated() / 1024**3:.1f}GB")

# 绘制训练历史
def plot_training_history():
    """绘制训练历史图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('训练历史 - 集成优化版本', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(train_history['train_loss'], label='训练Loss', color='blue')
    axes[0, 0].plot(train_history['val_loss'], label='验证Loss', color='red')
    axes[0, 0].set_title('Loss变化')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(train_history['train_acc'], label='训练Acc', color='blue')
    axes[0, 1].plot(train_history['val_acc'], label='验证Acc', color='red')
    if enable_ensemble:
        axes[0, 1].plot(ensemble_history['ensemble_val_acc'], label='集成Acc', color='green', linewidth=2)
    axes[0, 1].set_title('准确率变化')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[0, 2].plot(train_history['train_auc'], label='训练AUC', color='blue')
    axes[0, 2].plot(train_history['val_auc'], label='验证AUC', color='red')
    if enable_ensemble:
        axes[0, 2].plot(ensemble_history['ensemble_val_auc'], label='集成AUC', color='green', linewidth=2)
    axes[0, 2].set_title('AUC变化')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Precision
    axes[1, 0].plot(train_history['train_precision'], label='训练Precision', color='blue')
    axes[1, 0].plot(train_history['val_precision'], label='验证Precision', color='red')
    axes[1, 0].set_title('精确率变化')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(train_history['train_recall'], label='训练Recall', color='blue')
    axes[1, 1].plot(train_history['val_recall'], label='验证Recall', color='red')
    axes[1, 1].set_title('召回率变化')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # F1 Score
    axes[1, 2].plot(train_history['train_f1'], label='训练F1', color='blue')
    axes[1, 2].plot(train_history['val_f1'], label='验证F1', color='red')
    if enable_ensemble:
        axes[1, 2].plot(ensemble_history['ensemble_val_f1'], label='集成F1', color='green', linewidth=2)
    axes[1, 2].set_title('F1分数变化')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1 Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('./models/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# 绘制训练历史
plot_training_history()

print("📊 训练历史图表已保存到 ./models/training_history.png")