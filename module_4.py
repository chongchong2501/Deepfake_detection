#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 第4段：模型定义
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

# =============================================================================
# 第4段：模型定义
# =============================================================================

# 改进的CNN特征提取器
class ImprovedCNNFeatureExtractor(nn.Module): 
    def __init__(self, pretrained=True, backbone='resnet50'):
        super(ImprovedCNNFeatureExtractor, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        else:
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
            
        # 移除最后的分类层
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 添加特征降维层
        self.feature_reducer = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_dim = 512
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.feature_reducer(features)
        return features

# 改进的注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        # 输出投影
        output = self.output(attended)
        
        # 全局平均池化
        global_attended = torch.mean(output, dim=1)
        
        return global_attended, attention_weights.mean(dim=1)

# 优化的深度伪造检测模型
class OptimizedDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=512, num_layers=3, dropout=0.3, backbone='resnet50'):
        super(OptimizedDeepfakeDetector, self).__init__()
        
        # 改进的CNN特征提取器
        self.cnn = ImprovedCNNFeatureExtractor(pretrained=True, backbone=backbone)
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=self.cnn.output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 多头注意力机制
        self.attention = MultiHeadAttention(hidden_dim * 2, num_heads=8)
        
        # 改进的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # CNN特征提取
        x = x.view(batch_size * seq_len, channels, height, width)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(features)
        
        # 多头注意力机制
        attended, attention_weights = self.attention(lstm_out)
        
        # 分类
        output = self.classifier(attended)
        
        return torch.sigmoid(output.squeeze()), attention_weights

# 集成模型
class EnsembleDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=512):
        super(EnsembleDeepfakeDetector, self).__init__()
        
        # 多个不同的模型
        self.model1 = OptimizedDeepfakeDetector(num_classes, hidden_dim, backbone='resnet50')
        self.model2 = OptimizedDeepfakeDetector(num_classes, hidden_dim//2, backbone='efficientnet')
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out1, _ = self.model1(x)
        out2, _ = self.model2(x)
        
        # 融合预测结果
        combined = torch.stack([out1, out2], dim=1)
        final_output = self.fusion(combined)
        
        return final_output.squeeze(), None

# 模型创建函数
def create_optimized_model(model_type='optimized', hidden_dim=512, backbone='resnet50'):
    """
    创建优化后的模型
    
    Args:
        model_type: 'optimized' 或 'ensemble'
        hidden_dim: 隐藏层维度
        backbone: CNN骨干网络
    """
    if model_type == 'ensemble':
        model = EnsembleDeepfakeDetector(num_classes=1, hidden_dim=hidden_dim)
    else:
        model = OptimizedDeepfakeDetector(
            num_classes=1, 
            hidden_dim=hidden_dim, 
            num_layers=3, 
            dropout=0.3,
            backbone=backbone
        )
    
    return model

print("✅ 模型定义完成")