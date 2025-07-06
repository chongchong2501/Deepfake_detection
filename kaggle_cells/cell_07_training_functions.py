# Cell 7: 训练和验证函数
# 所有import语句已移至cell_01_imports_and_setup.py

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """GPU优化的训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # 高级性能监控
    data_load_times = []
    compute_times = []
    gpu_memory_peak = 0
    
    # 启用GPU流水线优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    print(f"⏳ 开始加载第一个训练批次...")
    for batch_idx, (data, target) in enumerate(pbar):
        data_start_time = time.time()
        
        if batch_idx == 0:
            print(f"✅ 第一个批次加载完成，数据形状: {data.shape}")
            print(f"📊 数据类型: {data.dtype}, 设备: {data.device}")
        
        # 高效数据传输（检查数据是否已在GPU上）
        if not data.is_cuda:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
        
        # 确保数据传输完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        data_load_time = time.time() - data_start_time
        data_load_times.append(data_load_time)
        
        compute_start_time = time.time()
        
        # 更高效的梯度清零
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # 混合精度训练
            with autocast():
                output, _ = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        
        # 在GPU上计算准确率
        with torch.no_grad():
            probs = torch.sigmoid(output)
            predicted = (probs > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 批量收集预测结果
            all_preds.extend(probs.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())

        compute_time = time.time() - compute_start_time
        compute_times.append(compute_time)
        
        # GPU内存监控
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_peak = max(gpu_memory_peak, current_memory)
        
        # 显示详细性能信息
        if batch_idx % 20 == 0 and batch_idx > 0:
            avg_data_time = np.mean(data_load_times[-20:])
            avg_compute_time = np.mean(compute_times[-20:])
            gpu_util = avg_compute_time / (avg_data_time + avg_compute_time) * 100
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'GPU%': f'{gpu_util:.1f}',
                'Mem': f'{current_memory:.1f}GB',
                'DataT': f'{avg_data_time*1000:.1f}ms'
            })
        else:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 智能GPU缓存管理
        if batch_idx % 30 == 0 and torch.cuda.is_available():
            if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.9:
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # 性能统计
    avg_data_load_time = np.mean(data_load_times)
    avg_compute_time = np.mean(compute_times)
    gpu_utilization = avg_compute_time / (avg_data_load_time + avg_compute_time) * 100
    
    print(f"\n🚀 训练性能深度分析:")
    print(f"平均数据加载时间: {avg_data_load_time*1000:.2f}ms")
    print(f"平均计算时间: {avg_compute_time*1000:.2f}ms")
    print(f"数据加载/计算比率: {avg_data_load_time/avg_compute_time:.3f} (目标: <0.1)")
    print(f"GPU内存峰值使用: {gpu_memory_peak:.1f}GB")
    print(f"GPU利用率: {gpu_utilization:.1f}%")
    print(f"GPU利用率优化: {'✅ 优秀' if avg_data_load_time/avg_compute_time < 0.1 else '⚠️ 需优化'}")

    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except:
        auc_score = 0.0

    return avg_loss, accuracy, auc_score

def validate_epoch(model, val_loader, criterion, device, scaler=None):
    """GPU优化的验证一个epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            # 非阻塞数据传输到GPU
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # 混合精度推理
            with autocast():
                output, _ = model(data)
                loss = criterion(output, target)

            total_loss += loss.item()
            
            # 在GPU上计算准确率
            probs = torch.sigmoid(output)
            predicted = (probs > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 批量收集预测结果
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # 定期清理GPU缓存
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except:
        auc_score = 0.0

    return avg_loss, accuracy, auc_score

print("✅ 训练和验证函数定义完成")