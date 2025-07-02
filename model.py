import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

# 测试代码
def test_model():
    # 创建随机输入
    batch_size = 2
    seq_len = 30
    channels = 3
    height = 128
    width = 128
    
    x = torch.randn(batch_size, seq_len, channels, height, width)
    
    # 测试标准模型
    model = create_model(model_type='standard', device='cpu')
    output, attention_weights = model(x)
    print(f"标准模型输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 测试轻量级模型
    light_model = create_model(model_type='lightweight', device='cpu')
    light_output = light_model(x)
    print(f"轻量级模型输出形状: {light_output.shape}")

if __name__ == "__main__":
    test_model()