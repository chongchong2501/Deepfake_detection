#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度伪造检测模型训练脚本 - 本地RTX4070版本

使用方法:
    python train.py [--config CONFIG_FILE] [--resume CHECKPOINT_PATH]
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

from config import config
from data_processing import prepare_data
from dataset import create_data_loaders
from model import create_model
from utils import FocalLoss, EarlyStopping, set_random_seed, format_time
from training import Trainer
from memory_manager import MemoryManager, print_memory_info, get_memory_suggestions

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度伪造检测模型训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--data-dir', type=str, default=None, help='数据集目录路径')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录路径')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    return parser.parse_args()

def setup_directories(output_dir):
    """设置输出目录（已由 config.setup_environment() 处理，此函数保持兼容性）"""
    output_dir = Path(output_dir)
    
    # 注意：目录创建已由 config.setup_environment() 统一处理
    # 这里只确保输出目录存在（防止配置未正确调用的情况）
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def save_config(config_dict, output_dir):
    """保存配置到文件"""
    config_path = output_dir / 'config.json'
    
    # 转换配置为可序列化的格式
    serializable_config = {}
    for key, value in config_dict.items():
        if isinstance(value, Path):
            serializable_config[key] = str(value)
        elif hasattr(value, '__dict__'):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_config, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {config_path}")

def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置随机种子
    set_random_seed()
    
    # 加载配置文件（在命令行参数覆盖之前）
    if args.config:
        config.load_config(args.config)
    else:
        config.load_config()  # 使用默认配置文件
    
    # 更新配置（命令行参数优先级更高）
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.no_cuda:
        config.USE_CUDA = False
    
    # 设置调试模式
    if args.debug:
        config.NUM_EPOCHS = 2
        config.BATCH_SIZE = min(config.BATCH_SIZE, 4)
        print("🐛 调试模式已启用")
    
    # 设置环境
    config.setup_environment()
    device = config.get_device()
    
    # 设置输出目录
    output_dir = setup_directories(config.OUTPUT_DIR)
    
    # 保存配置
    save_config(vars(config), output_dir)
    
    print("="*60)
    print("🚀 深度伪造检测模型训练")
    print("="*60)
    config.print_config()
    print("="*60)
    
    # 显示内存状态和优化建议
    print("\n📊 训练前内存状态:")
    print_memory_info()
    suggestions = get_memory_suggestions()
    if len(suggestions) > 1 or "内存使用状况良好" not in suggestions[0]:
        print("\n💡 内存优化建议:")
        for suggestion in suggestions:
            print(f"   {suggestion}")
    
    # 数据准备
    print("\n📊 准备数据...")
    train_data, val_data, test_data = prepare_data(
        data_dir=config.DATA_DIR,
        max_videos_per_class=None,  # 使用配置文件中的值
        force_reprocess=False
    )
    
    print(f"训练集: {len(train_data)} 个样本")
    print(f"验证集: {len(val_data)} 个样本")
    print(f"测试集: {len(test_data)} 个样本")
    
    # 创建数据加载器
    print("\n🔄 创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data
    )
    
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = create_model()
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建损失函数
    criterion = FocalLoss()
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 创建学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )
    
    # 创建早停机制
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA
    )
    
    # 恢复训练（如果指定）
    start_epoch = 0
    if args.resume:
        print(f"\n🔄 从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"从第 {start_epoch} 轮开始继续训练")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping
    )
    
    # 开始训练（使用内存管理）
    print("\n🎯 开始训练（带智能内存管理）...")
    try:
        # 使用带内存管理的训练方法
        history = trainer.start_training_with_memory_management(
            num_epochs=config.NUM_EPOCHS - start_epoch,
            save_dir=output_dir / 'models'
        )
        
        # 保存训练历史
        history_path = output_dir / 'results' / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"\n📊 训练历史已保存到: {history_path}")
        
        # 保存最终模型
        final_model_path = output_dir / 'models' / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': vars(config),
            'history': history
        }, final_model_path)
        print(f"🎯 最终模型已保存到: {final_model_path}")
        
        # 显示训练后的内存状态
        print("\n📊 训练后内存状态:")
        print_memory_info()
        
        print("\n✅ 训练完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        
        # 保存中断时的模型
        interrupted_model_path = output_dir / 'models' / 'interrupted_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': vars(config),
            'history': trainer.history
        }, interrupted_model_path)
        print(f"💾 中断时的模型已保存到: {interrupted_model_path}")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)