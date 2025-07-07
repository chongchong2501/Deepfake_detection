#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度伪造检测模型评估脚本 - 本地RTX4070版本

使用方法:
    python evaluate.py --model MODEL_PATH [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

from config import config
from data_processing import prepare_data
from dataset import create_data_loaders
from model import create_model
from utils import FocalLoss, load_model_checkpoint, format_time
from training import Evaluator

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度伪造检测模型评估')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data-dir', type=str, default=None, help='数据集目录路径')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录路径')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--save-predictions', action='store_true', help='保存预测结果')
    parser.add_argument('--plot-results', action='store_true', help='生成结果图表')
    
    return parser.parse_args()

def setup_directories(output_dir):
    """设置输出目录"""
    output_dir = Path(output_dir)
    
    # 创建必要的目录
    directories = [
        output_dir,
        output_dir / 'results',
        output_dir / 'plots',
        output_dir / 'predictions'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def load_model_for_evaluation(model_path, device):
    """加载模型用于评估"""
    print(f"📦 加载模型: {model_path}")
    
    # 创建模型
    model = create_model()
    model = model.to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 模型权重加载成功")
        
        # 打印模型信息
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            print(f"模型配置: {model_config.get('backbone', 'Unknown')}")
        
        if 'epoch' in checkpoint:
            print(f"训练轮数: {checkpoint['epoch']}")
        
        if 'best_val_acc' in checkpoint:
            print(f"最佳验证准确率: {checkpoint['best_val_acc']:.2f}%")
        
        if 'best_val_auc' in checkpoint:
            print(f"最佳验证AUC: {checkpoint['best_val_auc']:.4f}")
    else:
        # 直接加载模型状态字典
        model.load_state_dict(checkpoint)
        print("✅ 模型权重加载成功（直接格式）")
    
    model.eval()
    return model

def save_evaluation_results(results, metrics, output_dir, model_path):
    """保存评估结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细指标
    metrics_path = output_dir / 'results' / f'evaluation_metrics_{timestamp}.json'
    
    # 准备可序列化的指标
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    # 添加元信息
    evaluation_info = {
        'timestamp': timestamp,
        'model_path': str(model_path),
        'evaluation_metrics': serializable_metrics,
        'performance': {
            'avg_inference_time_ms': results['avg_inference_time'] * 1000,
            'total_inference_time_s': results['total_inference_time'],
            'samples_per_second': len(results['predictions']) / results['total_inference_time']
        }
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_info, f, indent=2, ensure_ascii=False)
    
    print(f"📊 评估指标已保存到: {metrics_path}")
    
    # 保存预测结果
    predictions_path = output_dir / 'predictions' / f'predictions_{timestamp}.csv'
    predictions_df = pd.DataFrame({
        'target': results['targets'],
        'prediction': results['predictions'],
        'score': results['scores']
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"🎯 预测结果已保存到: {predictions_path}")
    
    return metrics_path, predictions_path

def print_evaluation_summary(metrics, results):
    """打印评估摘要"""
    print("\n" + "="*60)
    print("📊 评估结果摘要")
    print("="*60)
    
    # 基础指标
    print(f"准确率 (Accuracy):        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"平衡准确率 (Balanced):     {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"精确率 (Precision):       {metrics['precision']:.4f}")
    print(f"召回率 (Recall):          {metrics['recall']:.4f}")
    print(f"特异性 (Specificity):     {metrics['specificity']:.4f}")
    print(f"F1分数 (F1-Score):        {metrics['f1']:.4f}")
    print(f"AUC-ROC:                  {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:                   {metrics['auc_pr']:.4f}")
    print(f"负预测值 (NPV):           {metrics['npv']:.4f}")
    
    print("\n" + "-"*40)
    print("混淆矩阵:")
    cm = metrics['confusion_matrix']
    print(f"真负例 (TN): {metrics['tn']:4d}  |  假正例 (FP): {metrics['fp']:4d}")
    print(f"假负例 (FN): {metrics['fn']:4d}  |  真正例 (TP): {metrics['tp']:4d}")
    
    print("\n" + "-"*40)
    print("性能指标:")
    print(f"平均推理时间:            {results['avg_inference_time']*1000:.2f} ms/batch")
    print(f"总推理时间:              {format_time(results['total_inference_time'])}")
    print(f"处理速度:                {len(results['predictions'])/results['total_inference_time']:.1f} 样本/秒")
    
    print("="*60)

def main():
    """主评估函数"""
    args = parse_args()
    
    # 更新配置
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
    # 注意：如果未指定输出目录，将使用 config.py 中的默认 OUTPUT_DIR
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.no_cuda:
        config.USE_CUDA = False
    
    # 设置环境
    config.setup_environment()
    device = config.get_device()
    
    # 设置输出目录
    output_dir = setup_directories(config.OUTPUT_DIR)
    
    print("="*60)
    print("🔍 深度伪造检测模型评估")
    print("="*60)
    print(f"模型文件: {args.model}")
    print(f"数据目录: {config.DATA_DIR}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print("="*60)
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return 1
    
    # 数据准备
    print("\n📊 准备数据...")
    try:
        train_data, val_data, test_data = prepare_data(
            data_dir=config.DATA_DIR,
            force_reprocess=False
        )
        print(f"测试集: {len(test_data)} 个样本")
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return 1
    
    # 创建数据加载器
    print("\n🔄 创建数据加载器...")
    try:
        _, _, test_loader = create_data_loaders(
            train_data, val_data, test_data
        )
        print(f"测试批次: {len(test_loader)}")
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return 1
    
    # 加载模型
    try:
        model = load_model_for_evaluation(model_path, device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 1
    
    # 创建损失函数
    criterion = FocalLoss()
    
    # 创建评估器
    evaluator = Evaluator(model, test_loader, criterion)
    
    # 开始评估
    print("\n🎯 开始评估...")
    try:
        results = evaluator.evaluate()
        
        # 计算详细指标
        metrics = evaluator.calculate_metrics(
            results['predictions'],
            results['targets'],
            results['scores']
        )
        
        # 打印结果摘要
        print_evaluation_summary(metrics, results)
        
        # 保存结果
        metrics_path, predictions_path = save_evaluation_results(
            results, metrics, output_dir, model_path
        )
        
        # 生成图表
        if args.plot_results:
            print("\n📈 生成结果图表...")
            
            # 混淆矩阵
            cm_path = output_dir / 'plots' / f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            evaluator.plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
            
            # ROC和PR曲线
            curves_path = output_dir / 'plots' / f'roc_pr_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            evaluator.plot_roc_pr_curves(results['targets'], results['scores'], curves_path)
        
        print("\n✅ 评估完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 评估被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)