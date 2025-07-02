# 深度伪造检测 Kaggle 笔记本

# 导入必要的库
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.notebook import tqdm
import zipfile
import kaggle

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建必要的目录
os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# 下载FaceForensics++数据集
def download_dataset():
    print("正在下载FaceForensics++数据集...")
    # 使用Kaggle API下载数据集
    # 注意：需要先设置Kaggle API凭证
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('c23/faceforensics', path='./data', unzip=True)
    print("数据集下载完成！")

# 如果在Kaggle环境中，可以直接使用输入数据
if os.path.exists('/kaggle/input/faceforensics'):
    print("在Kaggle环境中，使用现有数据集")
    data_dir = '/kaggle/input/faceforensics'
else:
    # 如果不在Kaggle环境中，下载数据集
    download_dataset()
    data_dir = './data/faceforensics'

# 提取视频帧
def extract_frames(video_path, output_dir, frames_per_video=30, resize_dim=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 均匀采样帧
    if frame_count <= frames_per_video:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = np.linspace(0, frame_count-1, frames_per_video, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 调整大小
            frame = cv2.resize(frame, resize_dim)
            # 保存帧
            output_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_frame_{i:03d}.jpg")
            cv2.imwrite(output_path, frame)
    
    cap.release()

# 处理所有视频
def process_videos(data_dir, output_dir, frames_per_video=30):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理真实视频
    real_dir = os.path.join(data_dir, 'original_sequences', 'c23', 'videos')
    real_output_dir = os.path.join(output_dir, 'real')
    os.makedirs(real_output_dir, exist_ok=True)
    
    print("处理真实视频...")
    for video_file in tqdm(os.listdir(real_dir)):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(real_dir, video_file)
            video_output_dir = os.path.join(real_output_dir, video_file.split('.')[0])
            extract_frames(video_path, video_output_dir, frames_per_video)
    
    # 处理伪造视频 - Deepfakes
    fake_dir = os.path.join(data_dir, 'manipulated_sequences', 'Deepfakes', 'c23', 'videos')
    fake_output_dir = os.path.join(output_dir, 'fake')
    os.makedirs(fake_output_dir, exist_ok=True)
    
    print("处理伪造视频...")
    for video_file in tqdm(os.listdir(fake_dir)):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(fake_dir, video_file)
            video_output_dir = os.path.join(fake_output_dir, video_file.split('.')[0])
            extract_frames(video_path, video_output_dir, frames_per_video)

# 创建数据集CSV文件
def create_dataset_csv(frames_dir, output_csv):
    data = []
    
    # 处理真实视频帧
    real_dir = os.path.join(frames_dir, 'real')
    for video_dir in os.listdir(real_dir):
        video_frames_dir = os.path.join(real_dir, video_dir)
        if os.path.isdir(video_frames_dir):
            frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')]
            frame_files.sort()
            
            # 将帧路径和标签添加到数据列表
            video_data = {
                'video_id': video_dir,
                'frames': [os.path.join(video_frames_dir, f) for f in frame_files],
                'label': 0  # 0表示真实
            }
            data.append(video_data)
    
    # 处理伪造视频帧
    fake_dir = os.path.join(frames_dir, 'fake')
    for video_dir in os.listdir(fake_dir):
        video_frames_dir = os.path.join(fake_dir, video_dir)
        if os.path.isdir(video_frames_dir):
            frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')]
            frame_files.sort()
            
            # 将帧路径和标签添加到数据列表
            video_data = {
                'video_id': video_dir,
                'frames': [os.path.join(video_frames_dir, f) for f in frame_files],
                'label': 1  # 1表示伪造
            }
            data.append(video_data)
    
    # 保存为CSV文件
    df = pd.DataFrame({
        'video_id': [item['video_id'] for item in data],
        'frames': [item['frames'] for item in data],
        'label': [item['label'] for item in data]
    })
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # 保存训练集和验证集
    train_df.to_csv(os.path.join(output_csv, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_csv, 'val.csv'), index=False)
    
    print(f"数据集CSV文件已创建：{output_csv}")
    print(f"训练集大小：{len(train_df)}，验证集大小：{len(val_df)}")

# 视频数据集类
class DeepfakeVideoDataset(Dataset):
    def __init__(self, csv_file, transform=None, max_frames=30):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_data = self.data.iloc[idx]
        frames_paths = eval(video_data['frames'])  # 将字符串转换回列表
        label = video_data['label']
        
        # 加载帧
        frames = []
        for frame_path in frames_paths[:self.max_frames]:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB
            
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        # 如果帧数不足，用零填充
        if len(frames) < self.max_frames:
            zero_frame = torch.zeros_like(frames[0])
            frames.extend([zero_frame] * (self.max_frames - len(frames)))
        
        # 将帧堆叠成张量
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, label

# 处理数据
frames_dir = './data/processed_frames'
csv_dir = './data'

# 如果在Kaggle环境中，检查是否已有处理好的数据
if os.path.exists('/kaggle/input/faceforensics-processed'):
    print("使用预处理好的数据")
    frames_dir = '/kaggle/input/faceforensics-processed'
    # 复制CSV文件到当前目录
    if os.path.exists('/kaggle/input/faceforensics-processed/train.csv'):
        os.system('cp /kaggle/input/faceforensics-processed/train.csv ./data/')
        os.system('cp /kaggle/input/faceforensics-processed/val.csv ./data/')
else:
    # 处理视频
    process_videos(data_dir, frames_dir)
    
    # 创建数据集CSV文件
    create_dataset_csv(frames_dir, csv_dir)

# 模型定义
class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器，使用预训练的ResNet作为基础模型"""
    def __init__(self, pretrained=True, feature_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        # 使用预训练的ResNet18作为特征提取器
        resnet = models.resnet18(pretrained=pretrained)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # 添加一个投影层，将特征维度调整为指定维度
        self.projection = nn.Linear(512, feature_dim)
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        # 重塑输入以处理所有帧
        x = x.view(batch_size * seq_len, c, h, w)
        # 提取特征
        features = self.features(x)
        # 重塑特征
        features = features.view(features.size(0), -1)
        # 投影到指定维度
        features = self.projection(features)
        # 重塑回序列形式
        features = features.view(batch_size, seq_len, -1)
        return features

class AttentionLayer(nn.Module):
    """注意力层，用于关注重要帧"""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, hidden_dim]
        # 计算注意力权重
        attention_weights = F.softmax(self.attention(x), dim=1)
        # 应用注意力权重
        context = torch.sum(attention_weights * x, dim=1)
        return context, attention_weights

class DeepfakeDetector(nn.Module):
    """深度伪造检测模型，结合CNN和RNN"""
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, dropout=0.5, bidirectional=True):
        super(DeepfakeDetector, self).__init__()
        
        # CNN特征提取器
        self.feature_extractor = CNNFeatureExtractor(pretrained=True, feature_dim=input_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 注意力层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AttentionLayer(lstm_output_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, channels, height, width]
        
        # 提取CNN特征
        features = self.feature_extractor(x)
        
        # 通过LSTM处理序列
        lstm_out, _ = self.lstm(features)
        
        # 应用注意力机制
        context, attention_weights = self.attention(lstm_out)
        
        # 分类
        output = self.classifier(context)
        
        return output, attention_weights

# 轻量级模型版本，适用于资源受限的环境
class LightweightDeepfakeDetector(nn.Module):
    """轻量级深度伪造检测模型"""
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=1, dropout=0.3):
        super(LightweightDeepfakeDetector, self).__init__()
        
        # 使用MobileNetV2作为特征提取器
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        self.projection = nn.Linear(1280, input_dim)
        
        # GRU替代LSTM以减少参数
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # 重塑输入以处理所有帧
        x = x.view(batch_size * seq_len, c, h, w)
        
        # 提取特征
        x = self.features(x)
        x = x.mean([2, 3])  # 全局平均池化
        x = self.projection(x)
        
        # 重塑回序列形式
        x = x.view(batch_size, seq_len, -1)
        
        # 通过GRU处理序列
        _, h_n = self.gru(x)
        
        # 使用最后一个隐藏状态进行分类
        output = self.classifier(h_n.squeeze(0))
        
        return output

# 创建模型实例的函数
def create_model(model_type='standard', device='cuda'):
    """创建模型实例
    
    Args:
        model_type: 'standard'或'lightweight'
        device: 'cuda'或'cpu'
    
    Returns:
        model: 模型实例
    """
    if model_type == 'standard':
        model = DeepfakeDetector()
    elif model_type == 'lightweight':
        model = LightweightDeepfakeDetector()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model.to(device)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(train_loader, desc='训练')
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if isinstance(model.forward(inputs), tuple):
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
        
        outputs = outputs.squeeze()
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        
        # 收集预测和目标
        preds = (outputs > 0.5).float().cpu().numpy()
        predictions.extend(preds)
        targets.extend(labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
    
    # 计算平均损失和指标
    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='验证')
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            # 前向传播
            if isinstance(model.forward(inputs), tuple):
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            
            outputs = outputs.squeeze()
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            
            # 收集预测和目标
            preds = (outputs > 0.5).float().cpu().numpy()
            predictions.extend(preds)
            targets.extend(labels.cpu().numpy())
            scores.extend(outputs.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
    
    # 计算平均损失和指标
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    auc = roc_auc_score(targets, scores) if len(set(targets)) > 1 else 0.0
    
    return epoch_loss, accuracy, precision, recall, f1, auc

# 保存检查点
def save_checkpoint(model, optimizer, epoch, metrics, model_dir, model_type, is_best=False):
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # 保存最新检查点
    checkpoint_path = os.path.join(model_dir, f'{model_type}_checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 如果是最佳模型，也保存一份
    if is_best:
        best_model_path = os.path.join(model_dir, f'{model_type}_model_best.pth')
        torch.save(checkpoint, best_model_path)
        print(f"保存最佳模型到 {best_model_path}")

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
    plt.close()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot([m['accuracy'] for m in train_metrics], label='训练准确率')
    plt.plot([m['accuracy'] for m in val_metrics], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'accuracy_curve.png'))
    plt.close()
    
    # 绘制F1分数曲线
    plt.figure(figsize=(10, 5))
    plt.plot([m['f1'] for m in train_metrics], label='训练F1分数')
    plt.plot([m['f1'] for m in val_metrics], label='验证F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('训练和验证F1分数')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'f1_curve.png'))
    plt.close()

# 训练模型
def train_model(model_type='lightweight', num_epochs=10, batch_size=8, lr=0.0001, weight_decay=1e-5, num_frames=30):
    # 设置目录
    model_dir = './models'
    log_dir = './logs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = DeepfakeVideoDataset(
        csv_file=os.path.join('./data', 'train.csv'),
        transform=transform,
        max_frames=num_frames
    )
    
    val_dataset = DeepfakeVideoDataset(
        csv_file=os.path.join('./data', 'val.csv'),
        transform=transform,
        max_frames=num_frames
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = create_model(model_type=model_type, device=device)
    print(f"模型类型: {model_type}")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 初始化变量
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    # 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_metrics.append({
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1
        })
        
        val_metrics.append({
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_rec,
            'f1': val_f1,
            'auc': val_auc
        })
        
        # 打印指标
        print(f"训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # 检查是否是最佳模型
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # 保存检查点
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'val_auc': val_auc
        }
        save_checkpoint(model, optimizer, epoch, metrics, model_dir, model_type, is_best)
        
        # 绘制训练曲线
        plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, log_dir)
    
    print("训练完成！")
    return model, train_losses, val_losses, train_metrics, val_metrics

# 评估函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='评估')
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            # 前向传播
            if isinstance(model.forward(inputs), tuple):
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            
            outputs = outputs.squeeze()
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            
            # 收集预测和目标
            preds = (outputs > 0.5).float().cpu().numpy()
            all_predictions.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
    
    # 计算平均损失
    test_loss = running_loss / len(test_loader.dataset)
    
    return test_loss, all_predictions, all_targets, all_scores

# 计算并打印指标
def calculate_metrics(predictions, targets, scores):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    auc_score = roc_auc_score(targets, scores) if len(set(targets)) > 1 else 0.0
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(targets, predictions, target_names=['真实', '伪造']))
    
    # 计算混淆矩阵
    cm = confusion_matrix(targets, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': cm
    }

# 绘制混淆矩阵
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['真实', '伪造'], yticklabels=['真实', '伪造'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 绘制ROC曲线
def plot_roc_curve(targets, scores, save_path):
    fpr, tpr, _ = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 主函数
def main():
    # 设置参数
    model_type = 'lightweight'  # 'standard' 或 'lightweight'
    num_epochs = 10
    batch_size = 8
    lr = 0.0001
    weight_decay = 1e-5
    num_frames = 30
    
    # 训练模型
    model, train_losses, val_losses, train_metrics, val_metrics = train_model(
        model_type=model_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_frames=num_frames
    )
    
    # 加载最佳模型
    best_model_path = os.path.join('./models', f'{model_type}_model_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型: {best_model_path}")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据集
    test_dataset = DeepfakeVideoDataset(
        csv_file=os.path.join('./data', 'val.csv'),  # 使用验证集作为测试集
        transform=transform,
        max_frames=num_frames
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 定义损失函数
    criterion = nn.BCELoss()
    
    # 评估模型
    print("开始评估...")
    test_loss, predictions, targets, scores = evaluate_model(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}")
    
    # 计算指标
    metrics = calculate_metrics(predictions, targets, scores)
    
    # 创建结果目录
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    print(f"混淆矩阵已保存到 {cm_path}")
    
    # 绘制ROC曲线
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plot_roc_curve(targets, scores, roc_path)
    print(f"ROC曲线已保存到 {roc_path}")
    
    # 保存评估结果
    results = {
        'test_loss': test_loss,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc']
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(results_dir, 'evaluation_results.csv'), index=False)
    print(f"评估结果已保存到 {os.path.join(results_dir, 'evaluation_results.csv')}")
    
    print("评估完成！")

# 执行主函数
if __name__ == "__main__":
    main()