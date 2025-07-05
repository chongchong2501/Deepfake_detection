# Cell 5: 模型定义

class OptimizedDeepfakeDetector(nn.Module):
    """优化的深度伪造检测模型"""
    
    def __init__(self, backbone='resnet50', hidden_dim=512, num_layers=2, 
                 dropout=0.3, use_attention=True):
        super(OptimizedDeepfakeDetector, self).__init__()
        
        self.use_attention = use_attention
        
        # 特征提取器
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的backbone: {backbone}")
        
        # 时序建模
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # 双向LSTM
        
        # 注意力机制
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # 分类器 (移除 Sigmoid，因为使用 BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames = x.shape[:2]
        
        # 重塑为 (batch_size * num_frames, channels, height, width)
        x = x.view(-1, *x.shape[2:])
        
        # 特征提取
        features = self.backbone(x)  # (batch_size * num_frames, feature_dim)
        
        # 重塑回时序格式
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(features)  # (batch_size, num_frames, hidden_dim*2)
        
        # 注意力机制
        attention_weights = None
        if self.use_attention:
            attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # 全局平均池化
            pooled = attended_out.mean(dim=1)  # (batch_size, hidden_dim*2)
        else:
            # 简单的全局平均池化
            pooled = lstm_out.mean(dim=1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output.squeeze(-1), attention_weights

print("✅ 模型定义完成")