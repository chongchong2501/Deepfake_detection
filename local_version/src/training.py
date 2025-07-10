# 训练和评估模块 - 本地RTX4070优化版本

import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import config
from utils import (
    calculate_accuracy, calculate_auc, PerformanceMonitor,
    save_model_checkpoint, print_gpu_memory_info, cleanup_gpu_memory
)
from memory_manager import MemoryManager, auto_memory_management

class Trainer:
    """训练器类 - RTX4070优化版本"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, early_stopping=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.device = config.get_device()
        
        # 混合精度训练
        self.use_amp = config.USE_MIXED_PRECISION
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 性能监控
        self.monitor = PerformanceMonitor()
        
        # 内存管理器
        self.memory_manager = MemoryManager(
            gpu_memory_threshold=0.85,
            cpu_memory_threshold=0.80,
            auto_cleanup_interval=30.0,
            enable_monitoring=True
        )
        
        # 注册内存清理回调
        self.memory_manager.register_cleanup_callback(self._cleanup_training_cache)
        self.memory_manager.register_warning_callback(self._memory_warning_handler)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_auc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'lr': []
        }
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} - Training', leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            # 数据传输到GPU
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # 梯度清零
            self.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output, _ = self.model(data)
                    loss = self.criterion(output, target)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            with torch.no_grad():
                probs = torch.sigmoid(output)
                predicted = (probs > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 收集预测结果
                all_preds.extend(probs.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())
            
            # 更新进度条
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
            # 定期清理GPU缓存和内存监控
            if batch_idx % 50 == 0:
                self.memory_manager.smart_cleanup()
                self.monitor.update()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        auc_score = calculate_auc(torch.tensor(all_preds), torch.tensor(all_targets))
        
        return avg_loss, accuracy, auc_score
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} - Validation', leave=False)
            
            for batch_idx, (data, target) in enumerate(pbar):
                # 数据传输到GPU
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # 前向传播
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output, _ = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output, _ = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # 计算准确率
                probs = torch.sigmoid(output)
                predicted = (probs > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 收集预测结果
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # 更新进度条
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
                
                # 定期清理GPU缓存
                if batch_idx % 50 == 0:
                    self.memory_manager.cleanup_gpu_memory()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        auc_score = calculate_auc(torch.tensor(all_preds), torch.tensor(all_targets))
        
        return avg_loss, accuracy, auc_score
    
    def train(self, num_epochs, save_dir=None):
        """完整训练流程"""
        print(f"开始训练 - {num_epochs} epochs")
        print(f"设备: {self.device}")
        print(f"混合精度: {self.use_amp}")
        print_gpu_memory_info()
        
        self.monitor.reset()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc, train_auc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_auc = self.validate_epoch(epoch)
            
            # 学习率调度
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_auc'].append(train_auc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, AUC: {train_auc:.4f}")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  耗时: {epoch_time:.1f}秒")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    best_model_path = save_dir / 'best_model.pth'
                    save_model_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, val_loss, val_acc, val_auc, best_model_path
                    )
            
            # 早停检查
            if self.early_stopping:
                if self.early_stopping(val_loss, self.model, epoch):
                    print(f"\n早停触发，在第{epoch+1}轮停止训练")
                    break
            
            # 内存监控和优化建议
            if epoch % 5 == 0:
                self.memory_manager.print_memory_report()
                suggestions = self.memory_manager.get_optimization_suggestions()
                if len(suggestions) > 1 or "内存使用状况良好" not in suggestions[0]:
                    print("💡 内存优化建议:")
                    for suggestion in suggestions:
                        print(f"   {suggestion}")
        
        # 训练完成统计
        stats = self.monitor.get_stats()
        print(f"\n训练完成!")
        print(f"总耗时: {stats['elapsed_time']:.1f}秒")
        print(f"平均CPU使用率: {stats['avg_cpu_usage']:.1f}%")
        print(f"峰值GPU内存: {stats['peak_gpu_memory_gb']:.2f}GB")
        
        return self.history
    
    def _cleanup_training_cache(self):
        """训练缓存清理回调"""
        try:
            # 清理模型缓存
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
            
            # 清理优化器状态（谨慎使用）
            if hasattr(self.optimizer, 'zero_grad'):
                self.optimizer.zero_grad(set_to_none=True)
            
            # 清理数据加载器缓存
            if hasattr(self.train_loader.dataset, 'clear_cache'):
                self.train_loader.dataset.clear_cache()
            if hasattr(self.val_loader.dataset, 'clear_cache'):
                self.val_loader.dataset.clear_cache()
            
            print("🧹 训练缓存已清理")
        except Exception as e:
            print(f"训练缓存清理失败: {e}")
    
    def _memory_warning_handler(self, message: str):
        """内存警告处理回调"""
        print(f"⚠️ 内存警告: {message}")
        
        # 自动降低batch size（如果可能）
        if "GPU内存严重不足" in message:
            current_batch_size = self.train_loader.batch_size
            if current_batch_size > 1:
                print(f"🔧 建议将batch_size从{current_batch_size}降低到{current_batch_size//2}")
    
    def start_training_with_memory_management(self, num_epochs, save_dir=None):
        """带内存管理的训练流程"""
        with self.memory_manager:
            return self.train(num_epochs, save_dir)

class Evaluator:
    """评估器类"""
    
    def __init__(self, model, test_loader, criterion):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = config.get_device()
    
    def evaluate(self):
        """全面评估模型"""
        print("🚀 开始模型评估...")
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_scores = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc="评估进度")):
                data, target = data.to(self.device), target.to(self.device)
                
                # 记录推理时间
                start_time = time.time()
                output, attention_weights = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 计算损失
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 收集预测结果
                probs = torch.sigmoid(output)
                predictions = (probs > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        avg_inference_time = np.mean(inference_times)
        
        print(f"✅ 评估完成")
        print(f"平均损失: {avg_loss:.4f}")
        print(f"平均推理时间: {avg_inference_time*1000:.2f} ms/batch")
        
        return {
            'loss': avg_loss,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'scores': np.array(all_scores),
            'avg_inference_time': avg_inference_time,
            'total_inference_time': np.sum(inference_times)
        }
    
    def calculate_metrics(self, predictions, targets, scores):
        """计算全面的评估指标"""
        # 基础指标
        accuracy = accuracy_score(targets, predictions)
        balanced_acc = balanced_accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(targets, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # 特异性和负预测值
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # AUC指标
        try:
            auc_roc = roc_auc_score(targets, scores)
        except:
            auc_roc = 0.0
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(targets, scores)
            auc_pr = auc(recall_curve, precision_curve)
        except:
            auc_pr = 0.0
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'npv': npv,
            'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    
    def plot_confusion_matrix(self, cm, save_path):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建标签
        labels = np.array([[
            f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
            for j in range(cm.shape[1])
        ] for i in range(cm.shape[0])])
        
        # 绘制热图
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                    xticklabels=['真实', '伪造'],
                    yticklabels=['真实', '伪造'],
                    cbar_kws={'label': '样本数量'})
        
        plt.title('混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        
        # 添加统计信息
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        stats_text = f'准确率: {accuracy:.3f}\n精确率: {precision:.3f}\n召回率: {recall:.3f}\nF1分数: {f1:.3f}'
        plt.text(2.1, 0.5, stats_text, fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"混淆矩阵已保存到: {save_path}")
    
    def plot_roc_pr_curves(self, targets, scores, save_path):
        """绘制ROC和PR曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(targets, scores)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('假正率')
        ax1.set_ylabel('真正率')
        ax1.set_title('ROC曲线')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(targets, scores)
        pr_auc = auc(recall_curve, precision_curve)
        
        ax2.plot(recall_curve, precision_curve, color='darkgreen', lw=2,
                 label=f'PR曲线 (AUC = {pr_auc:.4f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('召回率')
        ax2.set_ylabel('精确率')
        ax2.set_title('精确率-召回率曲线')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ROC/PR曲线已保存到: {save_path}")