# Cell 9: 数据处理和准备 - 双T4 GPU优化版本

# 创建数据目录
os.makedirs('./data', exist_ok=True)

# 如果需要处理数据（首次运行）
if not os.path.exists('./data/train.csv'):
    print("📁 开始数据处理...")
    
    # 调用前面定义的数据处理函数 - 双T4 GPU优化配置
    data_list = process_videos_simple(BASE_DATA_DIR, max_real=1000, max_fake=1000, max_frames=12)
    
    if len(data_list) == 0:
        print("❌ 未找到数据，请检查数据路径")
        raise ValueError("数据路径错误或数据不存在")
    
    # 创建数据集分割
    train_data, val_data, test_data = create_dataset_split(data_list)
    
    # 保存数据集
    save_dataset_to_csv(train_data, './data/train.csv')
    save_dataset_to_csv(val_data, './data/val.csv')
    save_dataset_to_csv(test_data, './data/test.csv')
    
    print(f"训练集: {len(train_data)} 个样本")
    print(f"验证集: {len(val_data)} 个样本")
    print(f"测试集: {len(test_data)} 个样本")
    
    # 计算数据集规模
    total_samples = len(train_data) + len(val_data) + len(test_data)
    print(f"总样本数: {total_samples}")
    print(f"数据集规模: {'大型' if total_samples > 3000 else '中型' if total_samples > 1000 else '小型'}")
    
    # 估算训练时间
    estimated_time_per_epoch = total_samples * 0.5 / 60  # 假设每个样本0.5秒
    print(f"预估单轮训练时间: {estimated_time_per_epoch:.1f}分钟")
    
    # 显示假视频方法分布
    print("\n假视频方法分布统计:")
    fake_method_counts = {}
    for item in data_list:
        if item['label'] == 1:  # 假视频
            method = item['method']
            fake_method_counts[method] = fake_method_counts.get(method, 0) + 1
    
    for method, count in fake_method_counts.items():
        print(f"  {method}: {count} 个视频")
    
else:
    print("📊 数据集已存在，跳过数据处理步骤")
    # 读取现有数据集信息
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/val.csv')
    test_df = pd.read_csv('./data/test.csv')
    
    print(f"训练集: {len(train_df)} 个样本")
    print(f"验证集: {len(val_df)} 个样本")
    print(f"测试集: {len(test_df)} 个样本")
    
    # 显示数据分布
    print("\n数据标签分布:")
    print("训练集:", train_df['label'].value_counts().to_dict())
    print("验证集:", val_df['label'].value_counts().to_dict())
    print("测试集:", test_df['label'].value_counts().to_dict())

print("✅ 数据准备完成")