# Cell 9: 数据准备 - 三步优化专用版本

DATA_BASE_DIR = '/kaggle/input/ff-c23/FaceForensics++_C23'

print("📁 创建数据目录...")
os.makedirs('./data', exist_ok=True)
os.makedirs('./data/frames', exist_ok=True)  # 预提取帧存储目录

def pre_extract_all_frames(video_data, frames_dir='./data/frames'):
    """
    预提取所有视频帧到硬盘 - 三步优化的第一步
    
    Args:
        video_data: 视频数据列表
        frames_dir: 帧存储目录
    
    Returns:
        extracted_data: 包含预提取帧路径的数据列表
    """
    print(f"🎬 开始预提取所有视频帧到 {frames_dir}...")
    
    extracted_data = []
    total_videos = len(video_data)
    
    for idx, video_info in enumerate(tqdm(video_data, desc="预提取帧")):
        try:
            video_path = video_info['video_path']
            label = video_info['label']
            
            # 生成帧文件路径
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_file = os.path.join(frames_dir, f"{video_name}_frames.pt")
            
            # 检查是否已存在
            if os.path.exists(frame_file):
                extracted_data.append({
                    'frame_path': frame_file,
                    'label': label,
                    'original_video': video_path
                })
                continue
            
            # 使用GPU加速提取帧
            frames = extract_frames_memory_efficient(video_path, max_frames=MAX_FRAMES_PER_VIDEO)
            
            if len(frames) > 0:
                # 转换为tensor并保存
                frames_tensor = torch.stack([
                    torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    for frame in frames
                ])
                
                torch.save(frames_tensor, frame_file)
                
                extracted_data.append({
                    'frame_path': frame_file,
                    'label': label,
                    'original_video': video_path
                })
            else:
                print(f"⚠️ 跳过无效视频: {video_path}")
                
        except Exception as e:
            print(f"❌ 预提取失败 {video_path}: {e}")
            continue
    
    print(f"✅ 预提取完成: {len(extracted_data)}/{total_videos} 个视频")
    return extracted_data


# ==================== 配置参数 ====================
# 可自定义预处理视频数量
MAX_REAL_VIDEOS = 500      # 真实视频数量
MAX_FAKE_VIDEOS = 500      # 假视频数量
MAX_FRAMES_PER_VIDEO = 16  # 每个视频提取的帧数

# 真假视频比例建议
# 1:1 - 平衡数据集，适合大多数情况
# 1:2 - 轻微偏向假视频，提高假视频检测能力
# 1:3 - 中等偏向假视频，适合实际应用场景
# 1:6 - 强烈偏向假视频，模拟真实世界分布
REAL_FAKE_RATIO = "1:1"  # 当前比例

print("📁 创建数据目录...")
os.makedirs('./data', exist_ok=True)
os.makedirs('./data/frames', exist_ok=True)  # 预提取帧存储目录

print(f"📋 配置参数:")
print(f"   真实视频数量: {MAX_REAL_VIDEOS}")
print(f"   假视频数量: {MAX_FAKE_VIDEOS}")
print(f"   每视频帧数: {MAX_FRAMES_PER_VIDEO}")
print(f"   真假比例: {REAL_FAKE_RATIO}")
print(f"   预计总样本: {MAX_REAL_VIDEOS + MAX_FAKE_VIDEOS}")

video_data = process_videos_simple(
    base_data_dir=DATA_BASE_DIR,  # 使用动态检测的数据路径
    max_real=MAX_REAL_VIDEOS,      # 使用自定义真视频数量
    max_fake=MAX_FAKE_VIDEOS,      # 使用自定义假视频数量
    max_frames=MAX_FRAMES_PER_VIDEO
)

print(f"📊 原始视频数据: {len(video_data)} 个样本")

# 统计真假视频分布
real_count = sum(1 for item in video_data if item['label'] == 0)
fake_count = sum(1 for item in video_data if item['label'] == 1)
print(f"   真实视频: {real_count} 个")
print(f"   假视频: {fake_count} 个")

# 步骤1: 预提取所有帧
extracted_data = pre_extract_all_frames(video_data)

if len(extracted_data) == 0:
    raise ValueError("❌ 预提取帧失败，无法继续。请检查视频路径和格式。")

print(f"✅ 预提取帧完成: {len(extracted_data)} 个样本")

# 数据集分割
print("📊 分割数据集...")
train_data, val_data, test_data = create_dataset_split(
    extracted_data,  # 使用预提取的数据
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)

print(f"训练集: {len(train_data)} 样本")
print(f"验证集: {len(val_data)} 样本")
print(f"测试集: {len(test_data)} 样本")

# 保存数据集
print("💾 保存数据集...")
save_dataset_to_csv(train_data, './data/train.csv')
save_dataset_to_csv(val_data, './data/val.csv')
save_dataset_to_csv(test_data, './data/test.csv')

# 显示数据集统计
print("\n📈 数据集统计:")
for name, data in [("训练", train_data), ("验证", val_data), ("测试", test_data)]:
    real_count = sum(1 for item in data if item['label'] == 0)
    fake_count = sum(1 for item in data if item['label'] == 1)
    print(f"{name}集: 真实={real_count}, 伪造={fake_count}, 总计={len(data)}")

# 显示伪造视频方法分布
print("\n🎭 伪造视频方法分布:")
fake_methods = {}
for item in extracted_data:
    if item['label'] == 1:  # 伪造视频
        original_path = item['original_video']
        if 'Deepfakes' in original_path:
            method = 'Deepfakes'
        elif 'Face2Face' in original_path:
            method = 'Face2Face'
        elif 'FaceSwap' in original_path:
            method = 'FaceSwap'
        elif 'NeuralTextures' in original_path:
            method = 'NeuralTextures'
        else:
            method = 'Unknown'
        
        fake_methods[method] = fake_methods.get(method, 0) + 1

for method, count in fake_methods.items():
    print(f"  {method}: {count} 个视频")

print(f"\n✅ 数据准备完成！")
print(f"   📊 数据分布: 真实视频 {real_count} | 假视频 {fake_count}")
print(f"   📈 当前比例: {REAL_FAKE_RATIO}")
print(f"   🎯 数据集规模: {dataset_size} ({total_samples}样本)")
print(f"   🚀 可以开始训练了！")