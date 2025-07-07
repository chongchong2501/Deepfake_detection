#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度伪造检测模型推理脚本 - 本地RTX4070版本

使用方法:
    python inference.py --model MODEL_PATH --video VIDEO_PATH [--output OUTPUT_PATH]
    python inference.py --model MODEL_PATH --video-dir VIDEO_DIR [--output OUTPUT_DIR]
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
import time
import cv2
from tqdm import tqdm

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

from config import config
from model import create_model
from data_processing import VideoProcessor
from utils import get_transforms, format_time

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='深度伪造检测模型推理')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    
    # 输入选项（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='单个视频文件路径')
    input_group.add_argument('--video-dir', type=str, help='视频目录路径')
    
    parser.add_argument('--output', type=str, help='输出文件/目录路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值 (默认: 0.5)')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小 (默认: 1)')
    parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--save-frames', action='store_true', help='保存提取的帧')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    return parser.parse_args()

def load_model_for_inference(model_path, device):
    """加载模型用于推理"""
    print(f"📦 加载模型: {model_path}")
    
    # 创建模型
    model = create_model()
    model = model.to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_val_acc' in checkpoint:
            print(f"✅ 模型加载成功 (验证准确率: {checkpoint['best_val_acc']:.2f}%)")
        else:
            print("✅ 模型加载成功")
    else:
        # 直接加载模型状态字典
        model.load_state_dict(checkpoint)
        print("✅ 模型加载成功")
    
    model.eval()
    return model

def preprocess_video(video_path, processor, transforms):
    """预处理单个视频"""
    try:
        # 提取帧
        frames = processor.extract_frames_gpu_accelerated(video_path)
        
        if frames is None or len(frames) == 0:
            return None, "无法提取帧"
        
        # 应用变换
        processed_frames = []
        for frame in frames:
            if transforms:
                frame = transforms(frame)
            processed_frames.append(frame)
        
        # 转换为tensor
        video_tensor = torch.stack(processed_frames)
        
        return video_tensor, None
    
    except Exception as e:
        return None, str(e)

def predict_single_video(model, video_tensor, device, threshold=0.5):
    """预测单个视频"""
    model.eval()
    
    with torch.no_grad():
        # 添加批次维度
        video_tensor = video_tensor.unsqueeze(0).to(device)
        
        # 推理
        start_time = time.time()
        output, attention_weights = model(video_tensor)
        inference_time = time.time() - start_time
        
        # 计算概率和预测
        prob = torch.sigmoid(output).item()
        prediction = prob > threshold
        
        return {
            'probability': prob,
            'prediction': prediction,
            'label': 'FAKE' if prediction else 'REAL',
            'confidence': prob if prediction else (1 - prob),
            'inference_time': inference_time,
            'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None
        }

def process_single_video(args, model, processor, transforms, device):
    """处理单个视频文件"""
    video_path = Path(args.video)
    
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return 1
    
    print(f"\n🎬 处理视频: {video_path.name}")
    
    # 预处理视频
    print("📊 提取和预处理帧...")
    video_tensor, error = preprocess_video(video_path, processor, transforms)
    
    if video_tensor is None:
        print(f"❌ 视频预处理失败: {error}")
        return 1
    
    print(f"✅ 成功提取 {video_tensor.shape[0]} 帧")
    
    # 推理
    print("🔍 进行推理...")
    result = predict_single_video(model, video_tensor, device, args.threshold)
    
    # 打印结果
    print("\n" + "="*50)
    print("🎯 推理结果")
    print("="*50)
    print(f"视频文件: {video_path.name}")
    print(f"预测标签: {result['label']}")
    print(f"置信度: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"原始概率: {result['probability']:.4f}")
    print(f"推理时间: {result['inference_time']*1000:.2f} ms")
    print("="*50)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        
        # 准备结果数据
        result_data = {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'prediction': result['label'],
            'probability': result['probability'],
            'confidence': result['confidence'],
            'threshold': args.threshold,
            'inference_time_ms': result['inference_time'] * 1000,
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model
        }
        
        # 保存为JSON
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
        else:
            # 保存为CSV
            df = pd.DataFrame([result_data])
            df.to_csv(output_path, index=False)
        
        print(f"💾 结果已保存到: {output_path}")
    
    return 0

def process_video_directory(args, model, processor, transforms, device):
    """处理视频目录"""
    video_dir = Path(args.video_dir)
    
    if not video_dir.exists():
        print(f"❌ 视频目录不存在: {video_dir}")
        return 1
    
    # 查找视频文件
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
        video_files.extend(video_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"❌ 在目录中未找到视频文件: {video_dir}")
        return 1
    
    print(f"\n📁 找到 {len(video_files)} 个视频文件")
    
    # 处理所有视频
    results = []
    
    for video_path in tqdm(video_files, desc="处理视频"):
        if args.verbose:
            print(f"\n🎬 处理: {video_path.name}")
        
        # 预处理视频
        video_tensor, error = preprocess_video(video_path, processor, transforms)
        
        if video_tensor is None:
            if args.verbose:
                print(f"❌ 预处理失败: {error}")
            
            result_data = {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'prediction': 'ERROR',
                'probability': None,
                'confidence': None,
                'error': error,
                'threshold': args.threshold,
                'inference_time_ms': None,
                'timestamp': datetime.now().isoformat(),
                'model_path': args.model
            }
        else:
            # 推理
            result = predict_single_video(model, video_tensor, device, args.threshold)
            
            result_data = {
                'video_path': str(video_path),
                'video_name': video_path.name,
                'prediction': result['label'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'threshold': args.threshold,
                'inference_time_ms': result['inference_time'] * 1000,
                'timestamp': datetime.now().isoformat(),
                'model_path': args.model
            }
            
            if args.verbose:
                print(f"✅ {result['label']} (置信度: {result['confidence']:.3f})")
        
        results.append(result_data)
    
    # 统计结果
    successful_results = [r for r in results if r['prediction'] != 'ERROR']
    real_count = sum(1 for r in successful_results if r['prediction'] == 'REAL')
    fake_count = sum(1 for r in successful_results if r['prediction'] == 'FAKE')
    error_count = len(results) - len(successful_results)
    
    print("\n" + "="*60)
    print("📊 批量处理结果统计")
    print("="*60)
    print(f"总视频数: {len(results)}")
    print(f"成功处理: {len(successful_results)}")
    print(f"处理失败: {error_count}")
    print(f"预测为真实: {real_count}")
    print(f"预测为伪造: {fake_count}")
    
    if successful_results:
        avg_confidence = np.mean([r['confidence'] for r in successful_results])
        avg_inference_time = np.mean([r['inference_time_ms'] for r in successful_results])
        print(f"平均置信度: {avg_confidence:.3f}")
        print(f"平均推理时间: {avg_inference_time:.2f} ms")
    
    print("="*60)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n💾 批量结果已保存到: {output_path}")
    
    return 0

def main():
    """主推理函数"""
    args = parse_args()
    
    # 更新配置
    if args.no_cuda:
        config.USE_CUDA = False
    
    # 设置环境
    config.setup_environment()
    device = config.get_device()
    
    print("="*60)
    print("🔍 深度伪造检测模型推理")
    print("="*60)
    print(f"模型文件: {args.model}")
    print(f"设备: {device}")
    print(f"分类阈值: {args.threshold}")
    
    if args.video:
        print(f"输入视频: {args.video}")
    else:
        print(f"输入目录: {args.video_dir}")
    
    if args.output:
        print(f"输出路径: {args.output}")
    
    print("="*60)
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return 1
    
    # 加载模型
    try:
        model = load_model_for_inference(model_path, device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return 1
    
    # 创建视频处理器和变换
    processor = VideoProcessor()
    transforms = get_transforms(mode='test')
    
    # 根据输入类型处理
    try:
        if args.video:
            return process_single_video(args, model, processor, transforms, device)
        else:
            return process_video_directory(args, model, processor, transforms, device)
    
    except KeyboardInterrupt:
        print("\n⚠️ 推理被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)