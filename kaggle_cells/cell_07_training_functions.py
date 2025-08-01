# Cell 7: 训练函数 - 集成多任务学习和高级优化策略

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, 
                use_amp=False, gradient_clip=1.0, ensemble_mode=False):
    """
    训练一个epoch - 支持集成学习和多任务学习
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        scheduler: 学习率调度器
        use_amp: 是否使用混合精度训练
        gradient_clip: 梯度裁剪阈值
        ensemble_mode: 是否为集成模式
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    # 集成模式的损失统计
    if ensemble_mode:
        ensemble_losses = {
            'main': 0.0,
            'spatial': 0.0,
            'temporal': 0.0,
            'ensemble': 0.0
        }
    
    # 混合精度训练
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    progress_bar = tqdm(train_loader, desc="训练中", leave=False)
    
    # 添加训练开始的调试信息
    print(f"🔍 训练开始调试信息:")
    print(f"   - 数据加载器长度: {len(train_loader)}")
    print(f"   - 当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"   - 设备: {device}")
    print(f"   - 混合精度: {'启用' if use_amp else '禁用'}")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # 定期清理GPU内存
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 内存监控
        if batch_idx % 20 == 0 and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 批次 {batch_idx}: GPU内存 {memory_allocated:.1f}GB / {memory_reserved:.1f}GB")
        
        # 处理不同的数据格式
        if len(batch_data) == 3:
            # 包含额外特征
            videos, labels, additional_features = batch_data
            videos = videos.to(device)
            labels = labels.to(device)
            
            # 处理额外特征
            if additional_features and isinstance(additional_features, dict):
                for key, value in additional_features.items():
                    if isinstance(value, torch.Tensor):
                        additional_features[key] = value.to(device)
        else:
            # 标准格式
            videos, labels = batch_data
            videos = videos.to(device)
            labels = labels.to(device)
            additional_features = None
        
        optimizer.zero_grad()
        
        try:
            if use_amp:
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    if additional_features is not None:
                        outputs = model(videos, additional_features)
                    else:
                        outputs = model(videos)
                    
                    # 检查模型输出是否包含NaN
                    if isinstance(outputs, dict):
                        for key, output in outputs.items():
                            if torch.isnan(output).any() or torch.isinf(output).any():
                                print(f"⚠️ 批次 {batch_idx}: 模型输出 {key} 包含NaN/Inf")
                                raise ValueError(f"Model output {key} contains NaN/Inf")
                    else:
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print(f"⚠️ 批次 {batch_idx}: 模型输出包含NaN/Inf")
                            raise ValueError("Model output contains NaN/Inf")
                    
                    # 计算损失
                    if ensemble_mode and isinstance(outputs, dict):
                        # 集成模式的多任务损失
                        losses = {}
                        total_ensemble_loss = 0
                        
                        for key, pred in outputs.items():
                            if pred.dim() > 1:
                                pred = pred.squeeze(-1)
                            loss = criterion(pred, labels)
                            losses[key] = loss
                            
                            # 不同任务的权重
                            if key == 'ensemble':
                                weight = 0.5  # 集成预测权重最高
                            elif key == 'main':
                                weight = 0.3
                            else:
                                weight = 0.1  # 辅助任务权重较低
                            
                            total_ensemble_loss += weight * loss
                        
                        loss = total_ensemble_loss
                        pred_probs = torch.sigmoid(outputs['ensemble'])
                        
                        # 更新集成损失统计
                        for key, l in losses.items():
                            ensemble_losses[key] += l.item()
                    else:
                        # 标准模式
                        if outputs.dim() > 1:
                            outputs = outputs.squeeze(-1)
                        loss = criterion(outputs, labels)
                        pred_probs = torch.sigmoid(outputs)
                    
                    # 检查损失是否为NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ 批次 {batch_idx}: 损失为NaN/Inf，跳过此批次")
                        raise ValueError("Loss is NaN/Inf")
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准精度训练
                if additional_features is not None:
                    outputs = model(videos, additional_features)
                else:
                    outputs = model(videos)
                
                # 检查模型输出是否包含NaN
                if isinstance(outputs, dict):
                    for key, output in outputs.items():
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            print(f"⚠️ 批次 {batch_idx}: 模型输出 {key} 包含NaN/Inf")
                            raise ValueError(f"Model output {key} contains NaN/Inf")
                else:
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"⚠️ 批次 {batch_idx}: 模型输出包含NaN/Inf")
                        raise ValueError("Model output contains NaN/Inf")
                
                # 计算损失
                if ensemble_mode and isinstance(outputs, dict):
                    # 集成模式的多任务损失
                    losses = {}
                    total_ensemble_loss = 0
                    
                    for key, pred in outputs.items():
                        if pred.dim() > 1:
                            pred = pred.squeeze(-1)
                        loss_item = criterion(pred, labels)
                        losses[key] = loss_item
                        
                        # 不同任务的权重
                        if key == 'ensemble':
                            weight = 0.5
                        elif key == 'main':
                            weight = 0.3
                        else:
                            weight = 0.1
                        
                        total_ensemble_loss += weight * loss_item
                    
                    loss = total_ensemble_loss
                    pred_probs = torch.sigmoid(outputs['ensemble'])
                    
                    # 更新集成损失统计
                    for key, l in losses.items():
                        ensemble_losses[key] += l.item()
                else:
                    # 标准模式
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, labels)
                    pred_probs = torch.sigmoid(outputs)
                
                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ 批次 {batch_idx}: 损失为NaN/Inf，跳过此批次")
                    raise ValueError("Loss is NaN/Inf")
                
                loss.backward()
                
                # 梯度裁剪
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                optimizer.step()
            
            # 确保pred_probs是正确的张量格式
            if pred_probs.dim() > 1:
                pred_probs = pred_probs.squeeze(-1)
            
            # 计算准确率
            predictions = (pred_probs > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            
            # 更新统计
            total_loss += loss.item()
            total_samples += labels.size(0)
            
            # 添加成功批次的调试信息（仅前3个批次）
            if batch_idx < 3:
                print(f"🔍 批次 {batch_idx} 成功:")
                print(f"   - 损失值: {loss.item():.6f}")
                print(f"   - 样本数: {labels.size(0)}")
                print(f"   - 预测概率范围: [{pred_probs.min().item():.4f}, {pred_probs.max().item():.4f}]")
                print(f"   - 标签分布: {labels.sum().item()}/{labels.size(0)}")
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = correct_predictions / total_samples
            
            if ensemble_mode:
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.4f}',
                    'Ensemble': f'{ensemble_losses["ensemble"]/(batch_idx+1):.4f}'
                })
            else:
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
            
            # 每个批次后清理变量
            del videos, labels
            if additional_features is not None:
                del additional_features
            if 'outputs' in locals():
                del outputs
            if 'pred_probs' in locals():
                del pred_probs
            
        except Exception as e:
            print(f"⚠️ 训练批次 {batch_idx} 出错: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            # 添加调试信息
            print(f"🔍 调试信息 - 当前批次: {batch_idx}, 总样本数: {total_samples}, 总损失: {total_loss}")
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # 检查是否有有效的训练数据
    if total_samples == 0:
        print("⚠️ 警告: 没有成功处理任何训练批次!")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
    
    # 学习率调度
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total_loss / max(len(train_loader), 1))
        else:
            scheduler.step()
    
    # 返回训练结果
    avg_loss = total_loss / max(len(train_loader), 1)
    accuracy = correct_predictions / max(total_samples, 1)
    
    # 添加详细调试信息
    print(f"🔍 训练结果调试:")
    print(f"   - 总损失: {total_loss}")
    print(f"   - 数据加载器长度: {len(train_loader)}")
    print(f"   - 平均损失: {avg_loss}")
    print(f"   - 正确预测数: {correct_predictions}")
    print(f"   - 总样本数: {total_samples}")
    print(f"   - 准确率: {accuracy}")
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    if ensemble_mode:
        # 添加集成损失统计
        for key in ensemble_losses:
            results[f'{key}_loss'] = ensemble_losses[key] / len(train_loader)
    
    return results

def validate_epoch(model, val_loader, criterion, device, ensemble_mode=False):
    """
    验证一个epoch - 支持集成学习评估
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        ensemble_mode: 是否为集成模式
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    all_predictions = []
    all_labels = []
    
    # 集成模式的损失统计
    if ensemble_mode:
        ensemble_losses = {
            'main': 0.0,
            'spatial': 0.0,
            'temporal': 0.0,
            'ensemble': 0.0
        }
        ensemble_predictions = {
            'main': [],
            'spatial': [],
            'temporal': [],
            'ensemble': []
        }
    
    progress_bar = tqdm(val_loader, desc="验证中", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # 处理不同的数据格式
                if len(batch_data) == 3:
                    videos, labels, additional_features = batch_data
                    videos = videos.to(device)
                    labels = labels.to(device)
                    
                    # 处理额外特征
                    if additional_features and isinstance(additional_features, dict):
                        for key, value in additional_features.items():
                            if isinstance(value, torch.Tensor):
                                additional_features[key] = value.to(device)
                else:
                    videos, labels = batch_data
                    videos = videos.to(device)
                    labels = labels.to(device)
                    additional_features = None
                
                # 前向传播
                if additional_features is not None:
                    outputs = model(videos, additional_features)
                else:
                    outputs = model(videos)
                
                # 计算损失和预测
                if ensemble_mode and isinstance(outputs, dict):
                    # 集成模式
                    losses = {}
                    total_ensemble_loss = 0
                    
                    for key, pred in outputs.items():
                        if pred.dim() > 1:
                            pred = pred.squeeze(-1)
                        loss_item = criterion(pred, labels)
                        losses[key] = loss_item
                        
                        # 权重与训练时保持一致
                        if key == 'ensemble':
                            weight = 0.5
                        elif key == 'main':
                            weight = 0.3
                        else:
                            weight = 0.1
                        
                        total_ensemble_loss += weight * loss_item
                        
                        # 保存预测结果
                        pred_probs_item = torch.sigmoid(pred)
                        ensemble_predictions[key].extend(pred_probs_item.cpu().numpy())
                        ensemble_losses[key] += loss_item.item()
                    
                    loss = total_ensemble_loss
                    pred_probs = torch.sigmoid(outputs['ensemble'])
                else:
                    # 标准模式
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, labels)
                    pred_probs = torch.sigmoid(outputs)
                
                # 确保pred_probs是正确的张量格式
                if pred_probs.dim() > 1:
                    pred_probs = pred_probs.squeeze(-1)
                
                # 计算准确率
                predictions = (pred_probs > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                
                # 保存预测和标签用于详细评估
                all_predictions.extend(pred_probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 更新统计
                total_loss += loss.item()
                total_samples += labels.size(0)
                
                # 更新进度条
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_samples
                progress_bar.set_postfix({
                    'Val Loss': f'{avg_loss:.4f}',
                    'Val Acc': f'{accuracy:.4f}'
                })
                
            except Exception as e:
                print(f"⚠️ 验证批次 {batch_idx} 出错: {e}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                continue
    
    # 检查是否有有效的验证数据
    if total_samples == 0:
        print("⚠️ 警告: 没有成功处理任何验证批次!")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'predictions': [],
            'labels': []
        }
    
    # 计算最终指标
    avg_loss = total_loss / max(len(val_loader), 1)
    accuracy = correct_predictions / max(total_samples, 1)
    
    # 计算AUC等高级指标
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
        auc_score = roc_auc_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            np.array(all_predictions) > 0.5, 
            average='binary'
        )
    except Exception as e:
        print(f"⚠️ 计算高级指标失败: {e}")
        auc_score = 0.0
        precision = recall = f1 = 0.0
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    if ensemble_mode:
        # 添加集成评估结果
        for key in ensemble_losses:
            results[f'{key}_loss'] = ensemble_losses[key] / len(val_loader)
            results[f'{key}_predictions'] = ensemble_predictions[key]
        
        # 计算集成模型的AUC
        try:
            ensemble_auc = roc_auc_score(all_labels, ensemble_predictions['ensemble'])
            results['ensemble_auc'] = ensemble_auc
        except:
            results['ensemble_auc'] = 0.0
    
    return results

def train_ensemble_models(models, train_loader, val_loader, criterion, optimizers, 
                         device, num_epochs=10, schedulers=None, use_amp=False):
    """
    训练多个模型进行集成学习
    
    Args:
        models: 模型列表
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizers: 优化器列表
        device: 设备
        num_epochs: 训练轮数
        schedulers: 学习率调度器列表
        use_amp: 是否使用混合精度训练
    """
    ensemble_results = []
    
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        print(f"\n🚀 训练集成模型 {i+1}/{len(models)}")
        
        scheduler = schedulers[i] if schedulers else None
        model_results = {'train_history': [], 'val_history': []}
        
        best_val_auc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_results = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scheduler=scheduler, use_amp=use_amp
            )
            
            # 验证
            val_results = validate_epoch(model, val_loader, criterion, device)
            
            # 保存历史
            model_results['train_history'].append(train_results)
            model_results['val_history'].append(val_results)
            
            # 保存最佳模型
            if val_results['auc'] > best_val_auc:
                best_val_auc = val_results['auc']
                best_model_state = model.state_dict().copy()
            
            print(f"训练 - Loss: {train_results['loss']:.4f}, Acc: {train_results['accuracy']:.4f}")
            print(f"验证 - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}, AUC: {val_results['auc']:.4f}")
        
        # 加载最佳模型权重
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model_results['best_val_auc'] = best_val_auc
        ensemble_results.append(model_results)
        
        print(f"✅ 模型 {i+1} 训练完成，最佳验证AUC: {best_val_auc:.4f}")
    
    return ensemble_results

def ensemble_predict(models, data_loader, device, weights=None):
    """
    使用多个模型进行集成预测
    
    Args:
        models: 模型列表
        data_loader: 数据加载器
        device: 设备
        weights: 模型权重（如果为None则使用平均权重）
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    all_predictions = []
    all_labels = []
    
    # 设置所有模型为评估模式
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="集成预测中"):
            if len(batch_data) == 3:
                videos, labels, additional_features = batch_data
                videos = videos.to(device)
                labels = labels.to(device)
                
                if additional_features and isinstance(additional_features, dict):
                    for key, value in additional_features.items():
                        if isinstance(value, torch.Tensor):
                            additional_features[key] = value.to(device)
            else:
                videos, labels = batch_data
                videos = videos.to(device)
                labels = labels.to(device)
                additional_features = None
            
            # 收集所有模型的预测
            batch_predictions = []
            for model in models:
                if additional_features is not None:
                    outputs = model(videos, additional_features)
                else:
                    outputs = model(videos)
                
                if isinstance(outputs, dict):
                    # 集成模式，使用ensemble输出
                    pred = outputs['ensemble']
                else:
                    pred = outputs
                
                if pred.dim() > 1:
                    pred = pred.squeeze(-1)
                
                pred_probs = torch.sigmoid(pred)
                batch_predictions.append(pred_probs.cpu().numpy())
            
            # 加权平均
            ensemble_pred = np.average(batch_predictions, axis=0, weights=weights)
            all_predictions.extend(ensemble_pred)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

print("✅ 优化训练函数定义完成")