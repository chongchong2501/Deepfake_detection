# Cell 7: 训练和验证函数

from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """GPU优化的训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc='Training', leave=False)

    for batch_idx, (data, target) in enumerate(pbar):
        # 非阻塞数据传输到GPU
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
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

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # 定期清理GPU缓存
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

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