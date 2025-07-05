# Cell 14: 结果保存和总结

print("💾 保存实验结果...")
print("=" * 60)

# 准备保存的结果数据
results_summary = {
    'experiment_info': {
        'timestamp': datetime.now().isoformat(),
        'model_architecture': 'OptimizedDeepfakeDetector',
        'backbone': 'resnet18',
        'total_epochs': len(train_history['train_loss']),
        'best_epoch': best_epoch + 1 if 'best_epoch' in locals() else len(train_history['train_loss']),
        'early_stopping': True,
        'mixed_precision': torch.cuda.is_available()
    },
    'dataset_info': {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'batch_size': batch_size,
        'num_workers': 2
    },
    'training_config': {
        'optimizer': 'AdamW',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'loss_function': 'FocalLoss',
        'scheduler': 'ReduceLROnPlateau',
        'early_stopping_patience': 5
    },
    'final_metrics': {
        'test_loss': float(eval_results['loss']),
        'accuracy': float(metrics['accuracy']),
        'balanced_accuracy': float(metrics['balanced_accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'specificity': float(metrics['specificity']),
        'f1_score': float(metrics['f1']),
        'auc_roc': float(metrics['auc_roc']),
        'auc_pr': float(metrics['auc_pr']),
        'npv': float(metrics['npv'])
    },
    'confusion_matrix': {
        'tn': int(metrics['tn']),
        'fp': int(metrics['fp']),
        'fn': int(metrics['fn']),
        'tp': int(metrics['tp'])
    },
    'performance': {
        'avg_inference_time_ms': float(eval_results['avg_inference_time'] * 1000),
        'total_inference_time_s': float(eval_results['total_inference_time']),
        'samples_per_second': float(len(eval_results['targets']) / eval_results['total_inference_time'])
    },
    'training_history': {
        'train_loss': [float(x) for x in train_history['train_loss']],
        'train_acc': [float(x) for x in train_history['train_acc']],
        'train_auc': [float(x) for x in train_history['train_auc']],
        'val_loss': [float(x) for x in train_history['val_loss']],
        'val_acc': [float(x) for x in train_history['val_acc']],
        'val_auc': [float(x) for x in train_history['val_auc']],
        'learning_rates': [float(x) for x in train_history['lr']]
    },
    'class_specific_metrics': {
        'real_video_accuracy': float(real_accuracy),
        'fake_video_accuracy': float(fake_accuracy),
        'real_samples_count': int(real_samples),
        'fake_samples_count': int(fake_samples)
    }
}

# 保存结果到JSON文件
results_file = './results/experiment_results.json'
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print(f"✅ 实验结果已保存到: {results_file}")

# 保存训练历史到CSV
history_df = pd.DataFrame(train_history)
history_df.to_csv('./results/training_history.csv', index=False)
print("✅ 训练历史已保存到: ./results/training_history.csv")

# 保存预测结果
predictions_df = pd.DataFrame({
    'true_label': eval_results['targets'],
    'predicted_label': eval_results['predictions'],
    'prediction_score': eval_results['scores']
})
predictions_df.to_csv('./results/test_predictions.csv', index=False)
print("✅ 测试预测结果已保存到: ./results/test_predictions.csv")

# 生成实验报告
print("\n📋 生成实验报告...")
report = f"""
深度伪造检测模型实验报告
{'='*50}

实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型架构: OptimizedDeepfakeDetector (ResNet18 + LSTM + Attention)

数据集信息:
- 训练样本: {len(train_dataset):,}
- 验证样本: {len(val_dataset):,}
- 测试样本: {len(test_dataset):,}
- 批次大小: {batch_size}

训练配置:
- 优化器: AdamW (lr=1e-4, weight_decay=1e-4)
- 损失函数: Focal Loss (alpha=1, gamma=2)
- 学习率调度: ReduceLROnPlateau
- 早停机制: patience=5
- 混合精度训练: {'启用' if torch.cuda.is_available() else '禁用'}

最终性能指标:
- 准确率: {metrics['accuracy']*100:.2f}%
- 平衡准确率: {metrics['balanced_accuracy']*100:.2f}%
- 精确率: {metrics['precision']:.4f}
- 召回率: {metrics['recall']:.4f}
- F1分数: {metrics['f1']:.4f}
- AUC-ROC: {metrics['auc_roc']:.4f}
- AUC-PR: {metrics['auc_pr']:.4f}

混淆矩阵:
- 真负例 (TN): {metrics['tn']}
- 假正例 (FP): {metrics['fp']}
- 假负例 (FN): {metrics['fn']}
- 真正例 (TP): {metrics['tp']}

类别特定性能:
- 真实视频检测准确率: {real_accuracy*100:.2f}%
- 伪造视频检测准确率: {fake_accuracy*100:.2f}%

推理性能:
- 平均推理时间: {eval_results['avg_inference_time']*1000:.2f} ms/batch
- 处理速度: {len(eval_results['targets'])/eval_results['total_inference_time']:.1f} samples/s

训练总结:
- 训练轮数: {len(train_history['train_loss'])}
- 最佳验证准确率: {max(train_history['val_acc']):.2f}%
- 最佳验证AUC: {max(train_history['val_auc']):.4f}

文件输出:
- 模型权重: ./models/best_model.pth
- 训练历史图: ./results/training_history.png
- 混淆矩阵图: ./results/evaluation/confusion_matrix.png
- ROC/PR曲线图: ./results/evaluation/roc_pr_curves.png
- 分数分布图: ./results/evaluation/score_distribution.png
- 实验结果: ./results/experiment_results.json
- 训练历史: ./results/training_history.csv
- 预测结果: ./results/test_predictions.csv

{'='*50}
实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 保存报告
with open('./results/experiment_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ 实验报告已保存到: ./results/experiment_report.txt")

# 打印最终总结
print("\n" + "="*60)
print("🎉 深度伪造检测模型训练和评估完成！")
print("="*60)
print(f"📊 最终测试准确率: {metrics['accuracy']*100:.2f}%")
print(f"📊 AUC-ROC分数: {metrics['auc_roc']:.4f}")
print(f"📊 F1分数: {metrics['f1']:.4f}")
print(f"⚡ 推理速度: {len(eval_results['targets'])/eval_results['total_inference_time']:.1f} samples/s")
print("\n📁 所有结果文件已保存到 ./results/ 目录")
print("📁 最佳模型已保存到 ./models/best_model.pth")
print("\n✨ 实验成功完成！可以在Kaggle中查看所有生成的图表和结果文件。")
print("="*60)

# 显示文件结构
print("\n📂 生成的文件结构:")
print("""
./models/
  └── best_model.pth
./results/
  ├── experiment_results.json
  ├── experiment_report.txt
  ├── training_history.csv
  ├── training_history.png
  ├── test_predictions.csv
  └── evaluation/
      ├── confusion_matrix.png
      ├── roc_pr_curves.png
      └── score_distribution.png
""")

print("\n🚀 可以使用以下代码加载训练好的模型进行推理:")
print("""
# 加载模型
model = OptimizedDeepfakeDetector(...)
checkpoint = torch.load('./models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 进行推理 (注意: 模型输出 logits，需要应用 sigmoid 获得概率)
with torch.no_grad():
    logits, attention = model(video_tensor)
    probs = torch.sigmoid(logits)  # 转换为概率
    prediction = (probs > 0.5).float()
    confidence = probs.item()
""")

print("\n💡 提示: 在Kaggle中运行时，建议按顺序执行所有cell，确保数据路径正确设置。")
print("\n⚠️  重要: 模型输出的是 logits，使用时必须先应用 sigmoid 函数转换为概率值！")