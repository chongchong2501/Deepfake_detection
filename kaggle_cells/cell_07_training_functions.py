# Cell 7: 训练和验证函数

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Kaggle T4 GPU优化的训练函数"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        # 数据传输到GPU
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # 梯度清零
        optimizer.zero_grad(set_to_none=True)

        # 前向传播 - 支持混合精度
        if scaler is not None:
            with autocast():
                output, _ = model(data)
                loss = criterion(output, target)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32训练
            output, _ = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        
        # 计算准确率
        with torch.no_grad():
            probs = torch.sigmoid(output)
            predicted = (probs > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 收集预测结果
            all_preds.extend(probs.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # 定期清理GPU缓存
        if batch_idx % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except:
        auc_score = 0.0
    
    # 计算类别平衡指标
    all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
    all_targets_array = np.array(all_targets)
    
    # 计算每个类别的指标
    real_mask = all_targets_array == 0
    fake_mask = all_targets_array == 1
    
    if np.sum(real_mask) > 0:
        real_acc = np.mean(all_preds_binary[real_mask] == all_targets_array[real_mask])
    else:
        real_acc = 0.0
        
    if np.sum(fake_mask) > 0:
        fake_acc = np.mean(all_preds_binary[fake_mask] == all_targets_array[fake_mask])
    else:
        fake_acc = 0.0
    
    print(f"  类别准确率 - 真实: {real_acc*100:.1f}%, 伪造: {fake_acc*100:.1f}%")

    return avg_loss, accuracy, auc_score

def validate_epoch(model, val_loader, criterion, device, scaler=None):
    """Kaggle T4 GPU优化的验证函数"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            # 数据传输到GPU
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # 前向传播 - 支持混合精度
            if scaler is not None:
                with autocast():
                    output, _ = model(data)
                    loss = criterion(output, target)
            else:
                output, _ = model(data)
                loss = criterion(output, target)

            total_loss += loss.item()
            
            # 计算准确率
            probs = torch.sigmoid(output)
            predicted = (probs > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 收集预测结果
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
    
    # 计算类别平衡指标
    all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
    all_targets_array = np.array(all_targets)
    
    # 计算每个类别的指标
    real_mask = all_targets_array == 0
    fake_mask = all_targets_array == 1
    
    if np.sum(real_mask) > 0:
        real_acc = np.mean(all_preds_binary[real_mask] == all_targets_array[real_mask])
    else:
        real_acc = 0.0
        
    if np.sum(fake_mask) > 0:
        fake_acc = np.mean(all_preds_binary[fake_mask] == all_targets_array[fake_mask])
    else:
        fake_acc = 0.0
    
    print(f"  类别准确率 - 真实: {real_acc*100:.1f}%, 伪造: {fake_acc*100:.1f}%")

    return avg_loss, accuracy, auc_score

print("✅ 训练和验证函数定义完成")