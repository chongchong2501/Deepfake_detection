# 模型定义 - 本地RTX4070优化版本

import torch
import torch.nn as nn
import torchvision.models as models
from config import config

class OptimizedDeepfakeDetector(nn.Module):
    """优化的深度伪造检测模型 - RTX4070版本"""
    
    def __init__(self, backbone=None, hidden_dim=None, num_layers=None, 
                 dropout=None, use_attention=None):
        super(OptimizedDeepfakeDetector, self).__init__()
        
        # 使用配置文件的默认值
        self.backbone_name = backbone or config.BACKBONE
        self.hidden_dim = hidden_dim or config.HIDDEN_DIM
        self.num_layers = num_layers or config.NUM_LSTM_LAYERS
        self.dropout = dropout or config.DROPOUT
        self.use_attention = use_attention if use_attention is not None else config.USE_ATTENTION
        
        # 特征提取器
        if self.backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif self.backbone_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的backbone: {self.backbone_name}")
        
        # 时序建模 - 双向LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = self.hidden_dim * 2  # 双向LSTM
        
        # 多头注意力机制
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=8,
                dropout=self.dropout,
                batch_first=True
            )
            
            # 层归一化
            self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # 分类器 - 更深的网络结构
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout // 2),
            
            nn.Linear(self.hidden_dim // 4, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames = x.shape[:2]
        
        # 重塑为 (batch_size * num_frames, channels, height, width)
        x = x.view(-1, *x.shape[2:])
        
        # 特征提取
        with torch.cuda.amp.autocast(enabled=config.USE_MIXED_PRECISION):
            features = self.backbone(x)  # (batch_size * num_frames, feature_dim)
        
        # 重塑回时序格式
        features = features.view(batch_size, num_frames, -1)
        
        # 确保LSTM输入为float32类型
        features = features.float()
        
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(features)  # (batch_size, num_frames, hidden_dim*2)
        
        # 注意力机制
        attention_weights = None
        if self.use_attention:
            # 多头自注意力
            attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
            
            # 残差连接和层归一化
            attended_out = self.layer_norm(attended_out + lstm_out)
            
            # 全局平均池化
            pooled = attended_out.mean(dim=1)  # (batch_size, hidden_dim*2)
        else:
            # 简单的全局平均池化
            pooled = lstm_out.mean(dim=1)
        
        # 分类
        output = self.classifier(pooled)
        
        return output.squeeze(-1), attention_weights
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.backbone_name,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'use_attention': self.use_attention,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024**2
        }
    
    def freeze_backbone(self, freeze=True):
        """冻结或解冻backbone参数"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        print(f"Backbone参数已{'冻结' if freeze else '解冻'}")
    
    def unfreeze_backbone_layers(self, num_layers=2):
        """解冻backbone的最后几层"""
        # 获取backbone的所有层
        backbone_layers = list(self.backbone.children())
        
        # 冻结所有层
        self.freeze_backbone(True)
        
        # 解冻最后几层
        for layer in backbone_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"已解冻backbone的最后{num_layers}层")

def create_model(device=None):
    """创建模型实例"""
    if device is None:
        device = config.get_device()
    
    model = OptimizedDeepfakeDetector()
    model = model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"\n🤖 模型信息:")
    print(f"架构: {model_info['backbone']} + LSTM + {'Attention' if model_info['use_attention'] else 'No Attention'}")
    print(f"总参数数量: {model_info['total_params']:,}")
    print(f"可训练参数数量: {model_info['trainable_params']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.1f} MB")
    print(f"设备: {device}")
    
    return model

def test_model_forward(model, device=None):
    """测试模型前向传播"""
    if device is None:
        device = config.get_device()
    
    model.eval()
    with torch.no_grad():
        # 创建测试输入
        batch_size = 2
        num_frames = config.MAX_FRAMES
        channels, height, width = 3, *config.FRAME_SIZE
        
        test_input = torch.randn(batch_size, num_frames, channels, height, width, device=device)
        
        print(f"\n🔍 测试模型前向传播...")
        print(f"输入形状: {test_input.shape}")
        
        try:
            outputs, attention_weights = model(test_input)
            print(f"输出形状: {outputs.shape}")
            print(f"输出范围: [{outputs.min():.3f}, {outputs.max():.3f}]")
            
            if attention_weights is not None:
                print(f"注意力权重形状: {attention_weights.shape}")
            
            print("✅ 模型前向传播测试成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型前向传播测试失败: {e}")
            return False