# Cell 9: 数据处理和准备

# 如果需要处理数据（首次运行）
if not os.path.exists('./data/train.csv'):
    print("📁 开始数据处理...")
    data_list = process_videos_simple(BASE_DATA_DIR, max_videos_per_class=120, max_frames=16)
    
    if len(data_list) == 0:
        print("❌ 未找到数据，请检查数据路径")
        raise ValueError("数据路径错误或数据不存在")
    
    train_data, val_data, test_data = create_dataset_split(data_list)
    
    # 保存数据集
    save_dataset_to_csv(train_data, './data/train.csv')
    save_dataset_to_csv(val_data, './data/val.csv')
    save_dataset_to_csv(test_data, './data/test.csv')
    
    print(f"训练集: {len(train_data)} 个样本")
    print(f"验证集: {len(val_data)} 个样本")
    print(f"测试集: {len(test_data)} 个样本")
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
    print("\n原始数据分布:")
    print("训练集标签分布:")
    print(train_df['label'].value_counts())
    print("\n验证集标签分布:")
    print(val_df['label'].value_counts())
    print("\n测试集标签分布:")
    print(test_df['label'].value_counts())
    
    # 检查类别不平衡并进行重采样
    real_count = (train_df['label'] == 0).sum()
    fake_count = (train_df['label'] == 1).sum()
    imbalance_ratio = fake_count / real_count if real_count > 0 else float('inf')
    
    print(f"\n类别不平衡比例: {imbalance_ratio:.2f} (伪造/真实)")
    
    if imbalance_ratio > 2.0:  # 如果不平衡比例超过2:1
        print("🔄 检测到严重类别不平衡，进行数据重采样...")
        
        # 分离真实和伪造样本
        real_samples = train_df[train_df['label'] == 0]
        fake_samples = train_df[train_df['label'] == 1]
        
        # 计算需要过采样的数量（使比例接近1:2）
        target_real_count = fake_count // 2
        if target_real_count > real_count:
            # 过采样真实样本
            oversample_count = target_real_count - real_count
            oversampled_real = real_samples.sample(n=oversample_count, replace=True, random_state=42)
            
            # 合并重采样后的数据
            balanced_train_df = pd.concat([real_samples, oversampled_real, fake_samples], ignore_index=True)
            balanced_train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
            
            # 保存重采样后的训练集
            balanced_train_df.to_csv('./data/train_balanced.csv', index=False)
            
            print(f"重采样后训练集: {len(balanced_train_df)} 个样本")
            print("重采样后标签分布:")
            print(balanced_train_df['label'].value_counts())
            
            # 更新训练数据引用
            train_df = balanced_train_df
        else:
            print("真实样本数量已足够，无需过采样")
    else:
        print("类别分布相对平衡，无需重采样")

print("✅ 数据准备完成")