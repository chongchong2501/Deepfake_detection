import os
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import create_model

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='深度伪造检测推理')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lightweight'], help='模型类型')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--num_frames', type=int, default=30, help='采样帧数')
    parser.add_argument('--threshold', type=float, default=0.5, help='检测阈值')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--save_frames', action='store_true', help='是否保存关键帧')
    return parser.parse_args()

# 提取视频帧
def extract_frames(video_path, num_frames=30, resize_dim=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    print(f"视频信息: {frame_count}帧, {fps}FPS, 时长{duration:.2f}秒")
    
    # 均匀采样帧
    if frame_count <= num_frames:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
    
    frames = []
    original_frames = []
    
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="提取帧")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            original_frames.append(frame.copy())
            # 调整大小
            frame = cv2.resize(frame, resize_dim)
            # 转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    return frames, original_frames, frame_indices, fps

# 预处理帧
def preprocess_frames(frames, transform):
    processed_frames = []
    for frame in frames:
        processed = transform(frame)
        processed_frames.append(processed)
    
    # 堆叠成张量 [num_frames, channels, height, width]
    frames_tensor = torch.stack(processed_frames)
    
    # 添加批次维度 [1, num_frames, channels, height, width]
    frames_tensor = frames_tensor.unsqueeze(0)
    
    return frames_tensor

# 可视化结果
def visualize_results(original_frames, frame_indices, scores, attention_weights=None, threshold=0.5, output_dir=None, video_name=None):
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
    
    # 绘制分数曲线
    frame_numbers = frame_indices
    ax1.plot(frame_numbers, scores, 'b-', linewidth=2)
    ax1.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold})')
    ax1.set_xlabel('帧索引')
    ax1.set_ylabel('伪造分数')
    ax1.set_title('每帧伪造检测分数')
    ax1.grid(True)
    ax1.legend()
    
    # 如果有注意力权重，绘制热力图
    if attention_weights is not None:
        attention_weights = attention_weights.squeeze().cpu().numpy()
        ax1.plot(frame_numbers, attention_weights, 'g-', linewidth=1, alpha=0.7, label='注意力权重')
    
    # 选择要显示的关键帧（分数最高的几帧）
    num_key_frames = min(5, len(original_frames))
    top_indices = np.argsort(scores)[-num_key_frames:]
    
    # 显示关键帧
    ax2.axis('off')
    for i, idx in enumerate(top_indices):
        ax = fig.add_subplot(2, num_key_frames, num_key_frames + i + 1)
        frame = cv2.cvtColor(original_frames[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(frame)
        ax.set_title(f'帧 {frame_indices[idx]}\n分数: {scores[idx]:.4f}')
        ax.axis('off')
        
        # 根据分数设置边框颜色
        if scores[idx] > threshold:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    plt.tight_layout()
    
    # 保存图表
    if output_dir and video_name:
        plt.savefig(os.path.join(output_dir, f"{video_name}_analysis.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

# 保存关键帧
def save_key_frames(original_frames, frame_indices, scores, threshold, output_dir, video_name):
    # 创建输出目录
    frames_dir = os.path.join(output_dir, f"{video_name}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # 保存所有帧和对应的分数
    for i, (frame, score) in enumerate(zip(original_frames, scores)):
        frame_path = os.path.join(frames_dir, f"frame_{frame_indices[i]:04d}_{score:.4f}.jpg")
        cv2.imwrite(frame_path, frame)
    
    # 保存分数最高的5帧
    top_frames_dir = os.path.join(output_dir, f"{video_name}_top_frames")
    os.makedirs(top_frames_dir, exist_ok=True)
    
    num_key_frames = min(5, len(original_frames))
    top_indices = np.argsort(scores)[-num_key_frames:]
    
    for i, idx in enumerate(top_indices):
        frame_path = os.path.join(top_frames_dir, f"top_{i+1}_frame_{frame_indices[idx]:04d}_{scores[idx]:.4f}.jpg")
        cv2.imwrite(frame_path, original_frames[idx])
    
    print(f"关键帧已保存到 {frames_dir} 和 {top_frames_dir}")

# 生成报告
def generate_report(video_path, prediction, confidence, frame_scores, output_dir, video_name):
    report_path = os.path.join(output_dir, f"{video_name}_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("深度伪造检测报告\n")
        f.write("=================\n\n")
        f.write(f"视频文件: {video_path}\n")
        f.write(f"检测结果: {'伪造' if prediction else '真实'}\n")
        f.write(f"置信度: {confidence:.4f}\n\n")
        
        f.write("帧分析:\n")
        f.write("--------\n")
        f.write(f"分析帧数: {len(frame_scores)}\n")
        f.write(f"平均伪造分数: {np.mean(frame_scores):.4f}\n")
        f.write(f"最高伪造分数: {np.max(frame_scores):.4f}\n")
        f.write(f"最低伪造分数: {np.min(frame_scores):.4f}\n")
        f.write(f"伪造分数标准差: {np.std(frame_scores):.4f}\n\n")
        
        # 记录高分数帧
        high_score_frames = [(i, score) for i, score in enumerate(frame_scores) if score > 0.7]
        if high_score_frames:
            f.write("高伪造分数帧 (>0.7):\n")
            for idx, score in sorted(high_score_frames, key=lambda x: x[1], reverse=True):
                f.write(f"  帧 {idx}: {score:.4f}\n")
    
    print(f"检测报告已保存到 {report_path}")

# 主函数
def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取视频名称（不带扩展名）
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 提取视频帧
    frames, original_frames, frame_indices, fps = extract_frames(
        args.video_path, args.num_frames
    )
    
    # 预处理帧
    frames_tensor = preprocess_frames(frames, transform)
    frames_tensor = frames_tensor.to(device)
    
    # 加载模型
    model = create_model(model_type=args.model_type, device=device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"加载模型: {args.model_path}")
    
    # 推理
    print("开始推理...")
    with torch.no_grad():
        if args.model_type == 'standard':
            outputs, attention_weights = model(frames_tensor)
            outputs = outputs.squeeze()
            
            # 获取每帧的分数
            # 注意：这里我们使用注意力机制的输出来估计每帧的分数
            # 实际上，标准模型是对整个序列进行分类，而不是单独的帧
            frame_scores = attention_weights.squeeze().cpu().numpy() * outputs.item()
        else:
            outputs = model(frames_tensor)
            outputs = outputs.squeeze()
            
            # 对于轻量级模型，我们没有注意力权重
            # 这里我们使用一个简单的方法来估计每帧的分数
            attention_weights = None
            frame_scores = np.ones(len(frames)) * outputs.item()
    
    # 获取最终预测和置信度
    prediction = outputs.item() > args.threshold
    confidence = outputs.item() if prediction else 1 - outputs.item()
    
    # 打印结果
    result = "伪造" if prediction else "真实"
    print(f"\n检测结果: {result}")
    print(f"置信度: {confidence:.4f}")
    
    # 可视化结果
    if args.visualize:
        visualize_results(
            original_frames, frame_indices, frame_scores, 
            attention_weights, args.threshold, 
            args.output_dir, video_name
        )
    
    # 保存关键帧
    if args.save_frames:
        save_key_frames(
            original_frames, frame_indices, frame_scores, 
            args.threshold, args.output_dir, video_name
        )
    
    # 生成报告
    generate_report(
        args.video_path, prediction, confidence, 
        frame_scores, args.output_dir, video_name
    )
    
    print("推理完成！")

if __name__ == "__main__":
    main()