#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度伪造检测系统演示脚本 - 本地RTX4070版本

这个脚本展示了如何使用深度伪造检测系统的主要功能：
1. 数据准备和预处理
2. 模型训练（演示模式）
3. 模型评估
4. 视频推理

使用方法:
    python demo.py [--data-dir DATA_DIR] [--quick]
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import time

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

from config import config
from data_processing import prepare_data
from dataset import create_data_loaders
from model import create_model
from utils import FocalLoss, EarlyStopping, set_random_seed, count_parameters
from training import Trainer, Evaluator

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度伪造检测系统演示')
    parser.add_argument('--data-dir', type=str, 
                       default="e:\\program\\Deepfake\\dataset\\FaceForensics++_C23",
                       help='数据集目录路径')
    parser.add_argument('--quick', action='store_true', 
                       help='快速演示模式（使用更少的数据和epoch）')
    parser.add_argument('--skip-training', action='store_true',
                       help='跳过训练，仅演示数据处理和模型创建')
    parser.add_argument('--output-dir', type=str, default='./demo_outputs',
                       help='输出目录')
    
    return parser.parse_args()

def print_section_header(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n📋 步骤 {step_num}: {title}")
    print("-" * 40)

def demo_data_preparation(data_dir, quick_mode=False):
    """演示数据准备过程"""
    print_section_header("数据准备和预处理演示")
    
    print_step(1, "检查数据集")
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 数据集目录不存在: {data_path}")
        print("请确保FaceForensics++数据集已下载到指定目录")
        return None, None, None
    
    # 检查FaceForensics++数据集结构
    original_dir = data_path / 'original'
    fake_methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    
    if original_dir.exists():
        real_videos = list(original_dir.glob('*.mp4'))
        print(f"✅ 找到 {len(real_videos)} 个真实视频")
    else:
        print("❌ 未找到真实视频目录 (original)")
        return None, None, None
    
    # 检查伪造视频目录
    total_fake_videos = 0
    found_fake_methods = []
    for method in fake_methods:
        method_dir = data_path / method
        if method_dir.exists():
            method_videos = list(method_dir.glob('*.mp4'))
            total_fake_videos += len(method_videos)
            found_fake_methods.append(f"{method}({len(method_videos)})")
    
    if total_fake_videos > 0:
        print(f"✅ 找到 {total_fake_videos} 个伪造视频")
        print(f"   伪造方法: {', '.join(found_fake_methods)}")
    else:
        print("❌ 未找到伪造视频目录")
        return None, None, None
    
    print_step(2, "数据预处理")
    print("开始提取视频帧和预处理...")
    
    # 在快速模式下限制视频数量
    max_videos = 20 if quick_mode else None
    
    try:
        train_data, val_data, test_data = prepare_data(
            data_dir=data_path,
            max_videos_per_class=max_videos,
            force_reprocess=False
        )
        
        print(f"✅ 数据预处理完成")
        print(f"   训练集: {len(train_data)} 个样本")
        print(f"   验证集: {len(val_data)} 个样本")
        print(f"   测试集: {len(test_data)} 个样本")
        
        return train_data, val_data, test_data
        
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        return None, None, None

def demo_model_creation():
    """演示模型创建"""
    print_section_header("模型创建演示")
    
    print_step(1, "创建模型")
    print(f"使用配置: {config.BACKBONE} + LSTM + 注意力机制")
    
    try:
        model = create_model()
        device = config.get_device()
        model = model.to(device)
        
        print(f"✅ 模型创建成功")
        print(f"   设备: {device}")
        
        # 统计参数
        params = count_parameters(model)
        print(f"   总参数: {params['total']:,}")
        print(f"   可训练参数: {params['trainable']:,}")
        
        # 测试前向传播
        print_step(2, "测试模型前向传播")
        model.eval()
        
        # 创建随机输入
        batch_size = 2
        num_frames = config.MAX_FRAMES
        height, width = config.FRAME_SIZE
        
        dummy_input = torch.randn(batch_size, num_frames, 3, height, width).to(device)
        
        with torch.no_grad():
            start_time = time.time()
            output, attention = model(dummy_input)
            inference_time = time.time() - start_time
        
        print(f"✅ 前向传播测试成功")
        print(f"   输入形状: {dummy_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   推理时间: {inference_time*1000:.2f} ms")
        
        if attention is not None:
            print(f"   注意力权重形状: {attention.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_training(model, train_data, val_data, quick_mode=False, output_dir='./demo_outputs'):
    """演示训练过程"""
    print_section_header("模型训练演示")
    
    print_step(1, "创建数据加载器")
    try:
        train_loader, val_loader, _ = create_data_loaders(train_data, val_data, val_data, quick_mode=quick_mode)
        print(f"✅ 数据加载器创建成功")
        print(f"   训练批次: {len(train_loader)}")
        print(f"   验证批次: {len(val_loader)}")
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return None
    
    print_step(2, "设置训练组件")
    
    # 创建损失函数
    criterion = FocalLoss()
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5 if quick_mode else config.NUM_EPOCHS
    )
    
    # 创建早停机制
    early_stopping = EarlyStopping(patience=3 if quick_mode else config.EARLY_STOPPING_PATIENCE)
    
    print("✅ 训练组件设置完成")
    
    print_step(3, "开始训练")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping
        )
        
        # 在快速模式下只训练几个epoch
        num_epochs = 3 if quick_mode else 10
        print(f"训练 {num_epochs} 个epoch（{'快速模式' if quick_mode else '演示模式'}）")
        
        history = trainer.train(num_epochs=num_epochs, save_dir=output_path)
        
        print("✅ 训练完成")
        print(f"   最终训练损失: {history['train_loss'][-1]:.4f}")
        print(f"   最终验证损失: {history['val_loss'][-1]:.4f}")
        print(f"   最终训练准确率: {history['train_acc'][-1]:.2f}%")
        print(f"   最终验证准确率: {history['val_acc'][-1]:.2f}%")
        
        return model, history
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def demo_evaluation(model, test_data):
    """演示模型评估"""
    print_section_header("模型评估演示")
    
    print_step(1, "准备测试数据")
    try:
        _, _, test_loader = create_data_loaders(test_data, test_data, test_data, quick_mode=True)
        print(f"✅ 测试数据加载器创建成功")
        print(f"   测试批次: {len(test_loader)}")
    except Exception as e:
        print(f"❌ 测试数据准备失败: {e}")
        return
    
    print_step(2, "模型评估")
    try:
        criterion = FocalLoss()
        evaluator = Evaluator(model, test_loader, criterion)
        
        results = evaluator.evaluate()
        metrics = evaluator.calculate_metrics(
            results['predictions'],
            results['targets'],
            results['scores']
        )
        
        print("✅ 评估完成")
        print(f"   准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   精确率: {metrics['precision']:.4f}")
        print(f"   召回率: {metrics['recall']:.4f}")
        print(f"   F1分数: {metrics['f1']:.4f}")
        print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"   平均推理时间: {results['avg_inference_time']*1000:.2f} ms/batch")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主演示函数"""
    args = parse_args()
    
    # 设置随机种子
    set_random_seed()
    
    # 更新配置
    config.DATA_DIR = Path(args.data_dir)
    config.OUTPUT_DIR = Path(args.output_dir)
    
    if args.quick:
        config.BATCH_SIZE = min(config.BATCH_SIZE, 4)
        print("🚀 快速演示模式已启用")
    
    # 设置环境
    config.setup_environment()
    
    print("="*60)
    print("🎬 深度伪造检测系统演示")
    print("="*60)
    print(f"数据目录: {config.DATA_DIR}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print(f"设备: {config.get_device()}")
    print(f"模式: {'快速演示' if args.quick else '完整演示'}")
    print("="*60)
    
    try:
        # 1. 数据准备演示
        train_data, val_data, test_data = demo_data_preparation(
            args.data_dir, quick_mode=args.quick
        )
        
        if train_data is None:
            print("❌ 数据准备失败，演示终止")
            return 1
        
        # 2. 模型创建演示
        model = demo_model_creation()
        
        if model is None:
            print("❌ 模型创建失败，演示终止")
            return 1
        
        # 3. 训练演示（可选）
        if not args.skip_training:
            model, history = demo_training(
                model, train_data, val_data, 
                quick_mode=args.quick, 
                output_dir=args.output_dir
            )
            
            if model is None:
                print("❌ 训练失败，跳过评估")
                return 1
        else:
            print_section_header("跳过训练演示")
            print("使用未训练的模型进行评估演示")
        
        # 4. 评估演示
        demo_evaluation(model, test_data)
        
        print_section_header("演示完成")
        print("🎉 深度伪造检测系统演示成功完成！")
        print("\n📚 接下来你可以：")
        print("   1. 使用 train.py 进行完整训练")
        print("   2. 使用 evaluate.py 进行详细评估")
        print("   3. 使用 inference.py 进行视频推理")
        print("   4. 查看 README.md 了解更多使用方法")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 演示被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)