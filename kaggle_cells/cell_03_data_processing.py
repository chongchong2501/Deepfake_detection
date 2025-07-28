# Cell 3: 数据处理函数
def extract_frames_gpu_accelerated(video_path, max_frames=16, target_size=(224, 224),
                                  quality_threshold=20, use_gpu=True, use_mtcnn=True):
    """GPU加速的帧提取函数 - 集成MTCNN人脸检测"""
    try:
        # 检查PyAV是否可用
        if not globals().get('PYAV_AVAILABLE', False):
            print(f"PyAV不可用，使用CPU回退处理: {video_path}")
            return extract_frames_cpu_fallback(video_path, max_frames, target_size, quality_threshold, use_mtcnn)
            
        # 设备选择 - 优先使用GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用torchvision的GPU加速视频读取
        if not use_gpu:
            device = torch.device('cpu')
            
        # 读取视频（torchvision自动处理解码）
        try:
            video_tensor, audio, info = read_video(video_path, pts_unit='sec')
            # video_tensor shape: (T, H, W, C)
        except Exception as e:
            print(f"GPU视频读取失败，回退到CPU: {e}")
            return extract_frames_cpu_fallback(video_path, max_frames, target_size, quality_threshold, use_mtcnn)
        
        if video_tensor.size(0) == 0:
            return []
            
        # 移动到GPU进行处理
        video_tensor = video_tensor.to(device, non_blocking=True)
        total_frames = video_tensor.size(0)
        
        # 智能帧采样策略
        if total_frames <= max_frames:
            frame_indices = torch.arange(0, total_frames, device=device)
        else:
            # 均匀采样
            step = total_frames / max_frames
            frame_indices = torch.arange(0, total_frames, step, device=device).long()[:max_frames]
        
        # 批量提取帧
        selected_frames = video_tensor[frame_indices]  # (max_frames, H, W, C)
        
        # GPU上进行质量检测（使用Sobel算子代替Laplacian）
        if quality_threshold > 0:
            # 转换为灰度图进行质量检测（先转换为float类型）
            gray_frames = selected_frames.float().mean(dim=-1, keepdim=True)  # (T, H, W, 1)
            gray_frames = gray_frames.permute(0, 3, 1, 2)  # (T, 1, H, W)
            
            # 使用Sobel算子计算图像质量
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32, device=device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32, device=device).view(1, 1, 3, 3)
            
            grad_x = F.conv2d(gray_frames, sobel_x, padding=1)
            grad_y = F.conv2d(gray_frames, sobel_y, padding=1)
            quality_scores = (grad_x.pow(2) + grad_y.pow(2)).mean(dim=[1, 2, 3])
            
            # 过滤低质量帧
            quality_mask = quality_scores > quality_threshold
            if quality_mask.sum() > 0:
                selected_frames = selected_frames[quality_mask]
            
        # GPU上进行尺寸调整
        selected_frames = selected_frames.permute(0, 3, 1, 2).float()  # (T, C, H, W)
        if selected_frames.size(-1) != target_size[0] or selected_frames.size(-2) != target_size[1]:
            selected_frames = F.interpolate(selected_frames, size=target_size, 
                                          mode='bilinear', align_corners=False)
        
        # 确保帧数足够
        current_frames = selected_frames.size(0)
        if current_frames < max_frames:
            # 重复最后一帧
            if current_frames > 0:
                last_frame = selected_frames[-1:].repeat(max_frames - current_frames, 1, 1, 1)
                selected_frames = torch.cat([selected_frames, last_frame], dim=0)
            else:
                # 创建黑色帧
                selected_frames = torch.zeros(max_frames, 3, target_size[0], target_size[1], 
                                            device=device, dtype=torch.float32)
        
        # 限制到最大帧数
        selected_frames = selected_frames[:max_frames]
        
        # 转换回CPU numpy格式（为了兼容现有代码）
        frames_cpu = selected_frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        frames_list = [frame for frame in frames_cpu]
        
        # 应用MTCNN人脸检测和裁剪
        if use_mtcnn and globals().get('MTCNN_AVAILABLE', False):
            frames_list = apply_mtcnn_face_detection(frames_list, target_size)
        
        return frames_list
        
    except Exception as e:
        print(f"GPU帧提取失败，回退到CPU: {e}")
        return extract_frames_cpu_fallback(video_path, max_frames, target_size, quality_threshold, use_mtcnn)

def apply_mtcnn_face_detection(frames, target_size=(224, 224)):
    """使用MTCNN进行人脸检测和裁剪 - 兼容新版本API"""
    try:
        # 新版本MTCNN构造函数不需要参数
        detector = MTCNN()
        processed_frames = []
        
        for frame in frames:
            # MTCNN需要RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            
            # 检测人脸 - 新版本API在detect_faces方法中传递参数
            results = detector.detect_faces(
                frame_rgb,
                min_face_size=40,  # 最小人脸尺寸
                threshold_pnet=0.6,  # PNet阈值
                threshold_rnet=0.7,  # RNet阈值  
                threshold_onet=0.8   # ONet阈值
            )
            
            if results and len(results) > 0:
                # 选择置信度最高的人脸
                best_face = max(results, key=lambda x: x['confidence'])
                
                if best_face['confidence'] > 0.9:  # 高置信度阈值
                    # 提取人脸区域
                    x, y, w, h = best_face['box']
                    
                    # 扩展边界框以包含更多上下文
                    margin = 0.2
                    x_margin = int(w * margin)
                    y_margin = int(h * margin)
                    
                    x1 = max(0, x - x_margin)
                    y1 = max(0, y - y_margin)
                    x2 = min(frame_rgb.shape[1], x + w + x_margin)
                    y2 = min(frame_rgb.shape[0], y + h + y_margin)
                    
                    # 裁剪人脸
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    
                    # 使用统一的帧处理函数
                    processed_frame = resize_and_validate_frame(face_crop, target_size, 0)  # MTCNN不需要额外质量检查
                    if processed_frame is None:
                        processed_frames.append(cv2.resize(face_crop, target_size))  # 如果处理失败，返回原帧
                    else:
                        processed_frames.append(processed_frame)
                else:
                    # 置信度不够，使用原始帧
                    processed_frames.append(cv2.resize(frame_rgb, target_size))
            else:
                # 没有检测到人脸，使用原始帧
                processed_frames.append(cv2.resize(frame_rgb, target_size))
        
        return processed_frames
        
    except Exception as e:
        print(f"MTCNN人脸检测失败，使用原始帧: {e}")
        return [cv2.resize(frame, target_size) for frame in frames]

def resize_and_validate_frame(frame, target_size, quality_threshold=20):
    """统一的帧处理函数：调整大小并验证质量"""
    if frame is None:
        return None
    
    # 调整尺寸
    resized_frame = cv2.resize(frame, target_size)
    
    # 质量检查
    if quality_threshold > 0:
        # 计算图像的方差作为质量指标
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY) if len(resized_frame.shape) == 3 else resized_frame
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < quality_threshold:
            return None
    
    return resized_frame

def extract_frames_cpu_fallback(video_path, max_frames=16, target_size=(224, 224), quality_threshold=20, use_mtcnn=True):
    """CPU回退的帧提取函数 - 集成MTCNN"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return frames

    # 均匀采样策略
    if total_frames <= max_frames:
        frame_indices = list(range(0, total_frames, max(1, total_frames // max_frames)))
    else:
        step = max(1, total_frames // max_frames)
        frame_indices = list(range(0, total_frames, step))[:max_frames]

    frame_count = 0
    for frame_idx in frame_indices:
        if frame_count >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 使用统一的帧处理函数
            processed_frame = resize_and_validate_frame(frame, target_size, quality_threshold)
            if processed_frame is None:
                continue
            frame = processed_frame
            frames.append(frame)
            frame_count += 1

    cap.release()

    # 如果帧数不足，重复最后一帧
    while len(frames) < max_frames and len(frames) > 0:
        frames.append(frames[-1].copy())

    # 应用MTCNN人脸检测
    if use_mtcnn and globals().get('MTCNN_AVAILABLE', False):
        frames = apply_mtcnn_face_detection(frames, target_size)

    return frames[:max_frames]

# 为了向后兼容，保留原函数名，但移除冗余参数
def extract_frames_memory_efficient(video_path, max_frames=16, target_size=(224, 224),
                                   quality_threshold=20, use_mtcnn=True):
    """兼容性包装函数，优先使用GPU加速，集成MTCNN
    注意：skip_frames参数已移除，因为GPU版本使用更智能的采样策略
    """
    return extract_frames_gpu_accelerated(video_path, max_frames, target_size, quality_threshold, use_mtcnn=use_mtcnn)

def process_videos_simple(base_data_dir, max_videos_per_class=60, max_frames=16, max_real=None, max_fake=None):
    """简化的视频处理函数 - 优化假视频平均分配"""
    # 打印设备信息（只打印一次）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 数据处理使用设备: {device}")
    
    # 向后兼容：如果指定了新参数，使用新参数；否则使用旧参数
    if max_real is None:
        max_real = max_videos_per_class
    if max_fake is None:
        max_fake = max_videos_per_class
    
    data_list = []
    fake_methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']

    print("开始处理真实视频...")
    # 处理真实视频
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
                frames = extract_frames_memory_efficient(video_path, max_frames)
                
                if len(frames) >= max_frames // 2:  # 至少要有一半的帧
                    data_list.append({
                        'video_path': video_path,
                        'frames': frames,
                        'label': 0,  # 真实视频
                        'method': 'original'
                    })
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {e}")
                continue

    # 处理伪造视频 - 平均分配策略
    print("开始处理伪造视频...")
    
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
        return data_list
    
    videos_per_method = max_fake // len(available_methods)
    remaining_videos = max_fake % len(available_methods)
    
    print(f"平均分配策略: 每种方法 {videos_per_method} 个视频")
    if remaining_videos > 0:
        print(f"剩余 {remaining_videos} 个视频将分配给前 {remaining_videos} 种方法")
    
    # 为每种方法采样视频
    selected_fake_videos = []
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
        
        selected_fake_videos.extend([(v, method) for v in method_selected])
    
    print(f"总共选择 {len(selected_fake_videos)} 个假视频进行处理")
    
    # 打乱选择的假视频顺序
    random.shuffle(selected_fake_videos)
    
    # 处理选择的假视频
    for video_path, method in tqdm(selected_fake_videos, desc="处理伪造视频"):
        try:
            frames = extract_frames_memory_efficient(video_path, max_frames)
            
            if len(frames) >= max_frames // 2:
                data_list.append({
                    'video_path': video_path,
                    'frames': frames,
                    'label': 1,  # 伪造视频
                    'method': method
                })
        except Exception as e:
            print(f"处理视频 {os.path.basename(video_path)} 时出错: {e}")
            continue

    # 统计最终结果
    method_counts = {}
    for item in data_list:
        if item['label'] == 1:  # 只统计假视频
            method = item['method']
            method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"\n✅ 数据处理完成，共处理 {len(data_list)} 个视频")
    print("假视频方法分布:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} 个视频")
    
    return data_list

def create_dataset_split(data_list, test_size=0.2, val_size=0.1):
    """创建数据集划分"""
    # 分离真实和伪造数据
    real_data = [item for item in data_list if item['label'] == 0]
    fake_data = [item for item in data_list if item['label'] == 1]
    
    print(f"真实视频: {len(real_data)} 个")
    print(f"伪造视频: {len(fake_data)} 个")
    
    # 分别划分真实和伪造数据
    real_train, real_temp = train_test_split(real_data, test_size=test_size+val_size, random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=test_size/(test_size+val_size), random_state=42)
    
    fake_train, fake_temp = train_test_split(fake_data, test_size=test_size+val_size, random_state=42)
    fake_val, fake_test = train_test_split(fake_temp, test_size=test_size/(test_size+val_size), random_state=42)
    
    # 合并数据
    train_data = real_train + fake_train
    val_data = real_val + fake_val
    test_data = real_test + fake_test
    
    # 打乱数据
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def save_dataset_to_csv(data_list, filename):
    """将数据集保存为CSV文件 - 支持预提取帧路径"""
    df_data = []
    for item in data_list:
        # 检查是否为预提取的帧数据
        if 'frame_path' in item:
            df_data.append({
                'frame_path': item['frame_path'],  # 预提取的帧路径
                'label': item['label'],
                'method': item['method'],
                'num_frames': item['num_frames']
            })
        else:
            # 向后兼容：原始视频路径格式
            df_data.append({
                'video_path': item['video_path'],
                'label': item['label'],
                'method': item['method'],
                'num_frames': len(item['frames'])
            })
    
    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"数据集已保存到: {filename}")
    return df

print("✅ 数据处理函数定义完成")