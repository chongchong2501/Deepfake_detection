# Cell 9: 数据准备 - 直接预提取优化版本

# ==================== 配置参数 ====================
# 数据集路径配置
DATA_BASE_DIR = '/kaggle/input/ff-c23/FaceForensics++_C23'

# 可自定义预处理视频数量
MAX_REAL_VIDEOS = 600      # 真实视频数量
MAX_FAKE_VIDEOS = 600      # 假视频数量
MAX_FRAMES_PER_VIDEO = 16  # 每个视频提取的帧数

# 真假视频比例建议
# 1:1 - 平衡数据集，适合大多数情况
# 1:2 - 轻微偏向假视频，提高假视频检测能力
# 1:3 - 中等偏向假视频，适合实际应用场景
# 1:6 - 强烈偏向假视频，模拟真实世界分布
REAL_FAKE_RATIO = "1:1"  # 当前比例

def direct_extract_frames_from_videos(base_data_dir, max_real=MAX_REAL_VIDEOS, max_fake=MAX_FAKE_VIDEOS, max_frames=MAX_FRAMES_PER_VIDEO, frames_dir='./data/frames'):
    """
    直接从视频目录预提取帧到硬盘 - 一步到位的优化方案
    
    Args:
        base_data_dir: 数据集根目录
        max_real: 最大真实视频数量
        max_fake: 最大假视频数量
        max_frames: 每个视频提取的帧数
        frames_dir: 帧存储目录
    
    Returns:
        extracted_data: 包含预提取帧路径的数据列表
    """
    print(f"🎬 开始直接预提取视频帧到 {frames_dir}...")
    
    # 创建必要的目录
    os.makedirs('./data', exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # 打印设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 数据处理使用设备: {device}")
    
    extracted_data = []
    fake_methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'DeepFakeDetection']
    
    # ==================== 处理真实视频 ====================
    print("🎯 开始处理真实视频...")
    original_dir = os.path.join(base_data_dir, 'original')
    if os.path.exists(original_dir):
        video_files = [f for f in os.listdir(original_dir)
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if len(video_files) > max_real:
            video_files = random.sample(video_files, max_real)
        
        print(f"找到 {len(video_files)} 个真实视频")
        
        for video_file in tqdm(video_files, desc="处理真实视频"):
            try:
                video_path = os.path.join(original_dir, video_file)
                
                # 生成帧文件路径
                video_name = os.path.splitext(video_file)[0]
                frame_file = os.path.join(frames_dir, f"{video_name}_frames.pt")
                
                # 检查是否已存在
                if os.path.exists(frame_file):
                    # 对于已存在的文件，我们需要加载它来获取帧数
                    try:
                        existing_frames = torch.load(frame_file)
                        num_frames = len(existing_frames)
                    except:
                        num_frames = max_frames  # 默认值
                    
                    extracted_data.append({
                        'frame_path': frame_file,
                        'label': 0,
                        'method': 'original',
                        'original_video': video_path,
                        'num_frames': num_frames
                    })
                    continue
                
                # 直接提取帧并保存
                frames = extract_frames_memory_efficient(video_path, max_frames)
                
                if len(frames) >= max_frames // 2:  # 至少要有一半的帧
                    # 转换为tensor并保存
                    frames_tensor = torch.stack([
                        torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                        for frame in frames
                    ])
                    
                    torch.save(frames_tensor, frame_file)
                    
                    extracted_data.append({
                        'frame_path': frame_file,
                        'label': 0,  # 真实视频
                        'method': 'original',
                        'original_video': video_path,
                        'num_frames': len(frames)
                    })
                else:
                    print(f"⚠️ 跳过帧数不足的视频: {video_file}")
                    
            except Exception as e:
                print(f"❌ 处理真实视频失败 {video_file}: {e}")
                continue
    
    # ==================== 处理假视频 - 平均分配策略 ====================
    print("🎭 开始处理假视频...")
    
    # 统计每种方法的可用视频数量
    method_videos = {}
    total_available_fake = 0
    
    for method in fake_methods:
        method_dir = os.path.join(base_data_dir, method)
        if os.path.exists(method_dir):
            videos = [os.path.join(method_dir, f) for f in os.listdir(method_dir) 
                     if f.endswith(('.mp4', '.avi', '.mov'))]
            method_videos[method] = videos
            total_available_fake += len(videos)
            print(f"  {method}: {len(videos)} 个视频")
        else:
            method_videos[method] = []
            print(f"  {method}: 目录不存在")
    
    print(f"总共可用假视频: {total_available_fake} 个")
    
    # 计算每种方法应该采样的视频数量（平均分配）
    available_methods = [method for method in fake_methods if len(method_videos[method]) > 0]
    if not available_methods:
        print("❌ 未找到任何假视频方法")
        return extracted_data
    
    videos_per_method = max_fake // len(available_methods)
    remaining_videos = max_fake % len(available_methods)
    
    print(f"平均分配策略: 每种方法 {videos_per_method} 个视频")
    if remaining_videos > 0:
        print(f"剩余 {remaining_videos} 个视频将分配给前 {remaining_videos} 种方法")
    
    # 为每种方法采样并直接处理视频
    for i, method in enumerate(available_methods):
        # 计算当前方法应该采样的数量
        current_method_quota = videos_per_method
        if i < remaining_videos:  # 前几种方法多分配一个
            current_method_quota += 1
        
        available_videos = method_videos[method]
        
        # 如果可用视频数量少于配额，全部使用
        if len(available_videos) <= current_method_quota:
            method_selected = available_videos
            print(f"  {method}: 使用全部 {len(method_selected)} 个视频")
        else:
            # 随机采样指定数量
            method_selected = random.sample(available_videos, current_method_quota)
            print(f"  {method}: 采样 {len(method_selected)} 个视频")
        
        # 直接处理选择的视频
        for video_path in tqdm(method_selected, desc=f"处理{method}"):
            try:
                # 生成帧文件路径
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_file = os.path.join(frames_dir, f"{video_name}_frames.pt")
                
                # 检查是否已存在
                if os.path.exists(frame_file):
                    # 对于已存在的文件，我们需要加载它来获取帧数
                    try:
                        existing_frames = torch.load(frame_file)
                        num_frames = len(existing_frames)
                    except:
                        num_frames = max_frames  # 默认值
                    
                    extracted_data.append({
                        'frame_path': frame_file,
                        'label': 1,
                        'method': method,
                        'original_video': video_path,
                        'num_frames': num_frames
                    })
                    continue
                
                # 直接提取帧并保存
                frames = extract_frames_memory_efficient(video_path, max_frames)
                
                if len(frames) >= max_frames // 2:
                    # 转换为tensor并保存
                    frames_tensor = torch.stack([
                        torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                        for frame in frames
                    ])
                    
                    torch.save(frames_tensor, frame_file)
                    
                    extracted_data.append({
                        'frame_path': frame_file,
                        'label': 1,  # 假视频
                        'method': method,
                        'original_video': video_path,
                        'num_frames': len(frames)
                    })
                else:
                    print(f"⚠️ 跳过帧数不足的视频: {os.path.basename(video_path)}")
                    
            except Exception as e:
                print(f"❌ 处理假视频失败 {os.path.basename(video_path)}: {e}")
                continue
    
    # 统计最终结果
    real_count = sum(1 for item in extracted_data if item['label'] == 0)
    fake_count = sum(1 for item in extracted_data if item['label'] == 1)
    
    method_counts = {}
    for item in extracted_data:
        if item['label'] == 1:  # 只统计假视频
            method = item['method']
            method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\n✅ 直接预提取完成: {len(extracted_data)} 个视频")
    print(f"   真实视频: {real_count} 个")
    print(f"   假视频: {fake_count} 个")
    print("假视频方法分布:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} 个视频")
    
    return extracted_data


print(f"📋 配置参数:")
print(f"   真实视频数量: {MAX_REAL_VIDEOS}")
print(f"   假视频数量: {MAX_FAKE_VIDEOS}")
print(f"   每视频帧数: {MAX_FRAMES_PER_VIDEO}")
print(f"   真假比例: {REAL_FAKE_RATIO}")
print(f"   预计总样本: {MAX_REAL_VIDEOS + MAX_FAKE_VIDEOS}")

# 直接预提取帧 - 一步到位的优化方案
extracted_data = direct_extract_frames_from_videos(
    base_data_dir=DATA_BASE_DIR,
    max_real=MAX_REAL_VIDEOS,
    max_fake=MAX_FAKE_VIDEOS,
    max_frames=MAX_FRAMES_PER_VIDEO
)

if len(extracted_data) == 0:
    raise ValueError("❌ 预提取帧失败，无法继续。请检查视频路径和格式。")

# 统计总体数据分布
total_real = sum(1 for item in extracted_data if item['label'] == 0)
total_fake = sum(1 for item in extracted_data if item['label'] == 1)
print(f"\n📊 总体数据统计: {len(extracted_data)} 个样本")
print(f"   真实视频: {total_real} 个")
print(f"   假视频: {total_fake} 个")

# 数据集分割
print("\n📊 分割数据集...")
train_data, val_data, test_data = create_dataset_split(
    extracted_data,  # 使用预提取的数据
    test_size=0.15,  # 测试集比例
    val_size=0.15    # 验证集比例
)

print(f"训练集: {len(train_data)} 样本")
print(f"验证集: {len(val_data)} 样本")
print(f"测试集: {len(test_data)} 样本")

# 保存数据集
print("\n💾 保存数据集...")
save_dataset_to_csv(train_data, './data/train.csv')
save_dataset_to_csv(val_data, './data/val.csv')
save_dataset_to_csv(test_data, './data/test.csv')

# 显示数据集统计
print("\n📈 数据集统计:")
for name, data in [("训练", train_data), ("验证", val_data), ("测试", test_data)]:
    real_count = sum(1 for item in data if item['label'] == 0)
    fake_count = sum(1 for item in data if item['label'] == 1)
    print(f"{name}集: 真实={real_count}, 伪造={fake_count}, 总计={len(data)}")

print(f"\n✅ 数据准备完成！")
print(f"   📊 数据分布: 真实视频 {total_real} | 假视频 {total_fake}")
print(f"   📈 当前比例: {REAL_FAKE_RATIO}")
print(f"   🎯 数据集规模: {len(extracted_data)} 个样本")
print(f"   🚀 可以开始训练了！")