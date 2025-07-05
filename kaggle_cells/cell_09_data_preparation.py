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
    print("\n数据分布:")
    print("训练集标签分布:")
    print(train_df['label'].value_counts())
    print("\n验证集标签分布:")
    print(val_df['label'].value_counts())
    print("\n测试集标签分布:")
    print(test_df['label'].value_counts())

print("✅ 数据准备完成")