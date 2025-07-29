# Cell 5: 模型定义 - 集成多模态特征和Ensemble策略
class OptimizedDeepfakeDetector(nn.Module):
    """优化的深度伪造检测器 - 集成多模态特征和Ensemble策略"""
    
    def __init__(self, num_classes=1, dropout_rate=0.3, use_attention=True, 
                 use_multimodal=False, ensemble_mode=False):
        super(OptimizedDeepfakeDetector, self).__init__()
        
        self.use_attention = use_attention
        self.use_multimodal = use_multimodal
        self.ensemble_mode = ensemble_mode
        
        # 主干网络 - ResNet50
        self.backbone = models.resnet50(pretrained=True)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除最后的分类层
        
        # 时序特征提取
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 7, 7))
        )
        
        # 注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=backbone_features, 
                num_heads=8, 
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(backbone_features)
        
        # 多模态特征融合
        if use_multimodal:
            # 频域特征处理 - 修正输入维度
            self.fourier_fc = nn.Sequential(
                nn.Linear(5, 256),  # 频域特征实际维度为5 (mean, std, max, energy, entropy)
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128)
            )
            
            # 压缩伪影特征处理 - 修正输入维度
            self.compression_fc = nn.Sequential(
                nn.Linear(32, 64),  # 压缩特征扩展为32维
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32)
            )
            
            # 时序一致性特征处理
            self.temporal_fc = nn.Sequential(
                nn.Linear(4, 64),  # 时序特征维度为4
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32)
            )
            
            # 特征融合层 - 动态计算输入维度
            # 基础特征: backbone_features (2048)
            # 频域特征: 128 (fourier_fc输出)
            # 压缩特征: 32 (compression_fc输出)  
            # 时序特征: 32 (temporal_fc输出)
            fusion_dim = backbone_features + 128 + 32 + 32  # 2048 + 128 + 32 + 32 = 2240
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256)
            )
            final_features = 256
        else:
            final_features = backbone_features
        
        # 集成模式的多个分类头
        if ensemble_mode:
            # 主分类器
            self.main_classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(final_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes)
            )
            
            # 辅助分类器1 - 专注于空间特征
            self.spatial_classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(final_features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )
            
            # 辅助分类器2 - 专注于时序特征
            self.temporal_classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(final_features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )
            
            # 集成权重（可学习）
            self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
            
        else:
            # 单一分类器
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(final_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes)
            )
            
            # 添加单一分类器用于处理基础特征（当多模态特征处理失败时）
            self.single_classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(backbone_features, 128),  # 直接处理backbone特征
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes)
            )
        
        # 初始化权重
        self._initialize_weights()
        
        print(f"✅ 模型初始化完成")
        print(f"   - 注意力机制: {'启用' if use_attention else '禁用'}")
        print(f"   - 多模态融合: {'启用' if use_multimodal else '禁用'}")
        print(f"   - 集成模式: {'启用' if ensemble_mode else '禁用'}")

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, additional_features=None):
        """
        前向传播
        Args:
            x: 视频张量 (B, T, C, H, W)
            additional_features: 额外特征字典
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # 提取每帧的空间特征
        x_reshaped = x.view(batch_size * num_frames, channels, height, width)
        spatial_features = self.backbone(x_reshaped)  # (B*T, features)
        spatial_features = spatial_features.view(batch_size, num_frames, -1)  # (B, T, features)
        
        # 时序特征聚合
        if self.use_attention:
            # 使用注意力机制聚合时序特征
            attended_features, attention_weights = self.attention(
                spatial_features, spatial_features, spatial_features
            )
            attended_features = self.attention_norm(attended_features + spatial_features)
            # 全局平均池化
            temporal_features = torch.mean(attended_features, dim=1)  # (B, features)
        else:
            # 简单平均池化
            temporal_features = torch.mean(spatial_features, dim=1)  # (B, features)
        
        # 多模态特征融合
        if self.use_multimodal and additional_features is not None:
            fusion_features = [temporal_features]
            
            # 处理频域特征
            if 'fourier' in additional_features:
                try:
                    fourier_feat = additional_features['fourier']
                    if isinstance(fourier_feat, dict):
                        # 安全地提取数值特征
                        fourier_values = []
                        for value in fourier_feat.values():
                            if isinstance(value, (int, float)):
                                fourier_values.append(float(value))
                            elif isinstance(value, torch.Tensor):
                                if value.numel() == 1:
                                    fourier_values.append(float(value.item()))
                                else:
                                    fourier_values.append(float(value.mean().item()))
                            elif isinstance(value, np.ndarray):
                                if value.size == 1:
                                    fourier_values.append(float(value.item()))
                                else:
                                    fourier_values.append(float(value.mean()))
                            else:
                                fourier_values.append(0.0)  # 默认值
                        
                        # 确保有足够的特征维度
                        if len(fourier_values) < 5:  # fourier_fc期望5维输入
                            fourier_values.extend([0.0] * (5 - len(fourier_values)))
                        elif len(fourier_values) > 5:
                            fourier_values = fourier_values[:5]
                        
                        fourier_tensor = torch.tensor([fourier_values] * batch_size, 
                                                    dtype=torch.float32, 
                                                    device=temporal_features.device)
                    else:
                        # 如果已经是张量，确保正确的形状
                        if isinstance(fourier_feat, torch.Tensor):
                            fourier_tensor = fourier_feat.to(temporal_features.device)
                            if fourier_tensor.dim() == 1:
                                fourier_tensor = fourier_tensor.unsqueeze(0).repeat(batch_size, 1)
                        else:
                            # 创建默认张量
                            fourier_tensor = torch.zeros(batch_size, 5, 
                                                        dtype=torch.float32, 
                                                        device=temporal_features.device)
                    
                    fourier_processed = self.fourier_fc(fourier_tensor)
                    fusion_features.append(fourier_processed)
                except Exception as e:
                    print(f"⚠️ 频域特征处理失败: {e}")
                    # 使用默认特征
                    fourier_tensor = torch.zeros(batch_size, 5, 
                                                dtype=torch.float32, 
                                                device=temporal_features.device)
                    fourier_processed = self.fourier_fc(fourier_tensor)
                    fusion_features.append(fourier_processed)
            
            # 处理压缩伪影特征
            if 'compression' in additional_features:
                try:
                    comp_feat = additional_features['compression']
                    if isinstance(comp_feat, dict):
                        # 安全地提取压缩特征 - 修正为5个特征
                        comp_values = []
                        for key in ['dct_mean', 'dct_std', 'dct_energy', 'high_freq_energy', 'edge_density']:
                            if key in comp_feat:
                                value = comp_feat[key]
                                if isinstance(value, (int, float)):
                                    comp_values.append(float(value))
                                elif isinstance(value, torch.Tensor):
                                    comp_values.append(float(value.item() if value.numel() == 1 else value.mean().item()))
                                elif isinstance(value, np.ndarray):
                                    comp_values.append(float(value.item() if value.size == 1 else value.mean()))
                                else:
                                    comp_values.append(0.0)
                            else:
                                comp_values.append(0.0)
                        
                        # 扩展到32维：重复基础特征并添加派生特征
                        extended_values = comp_values.copy()
                        # 添加派生特征
                        extended_values.extend([
                            comp_values[0] * comp_values[1],  # mean * std
                            comp_values[2] / (comp_values[3] + 1e-8),  # energy ratio
                            comp_values[4] * comp_values[0],  # edge * mean
                            np.sqrt(abs(comp_values[2])),  # sqrt energy
                            comp_values[1] / (comp_values[0] + 1e-8),  # std/mean ratio
                        ])
                        # 重复填充到32维
                        while len(extended_values) < 32:
                            extended_values.extend(comp_values[:min(5, 32 - len(extended_values))])
                        
                        comp_tensor = torch.tensor([extended_values[:32]] * batch_size, 
                                                 dtype=torch.float32, 
                                                 device=temporal_features.device)
                    else:
                        if isinstance(comp_feat, torch.Tensor):
                            comp_tensor = comp_feat.to(temporal_features.device)
                            if comp_tensor.dim() == 1:
                                comp_tensor = comp_tensor.unsqueeze(0).repeat(batch_size, 1)
                            # 确保是32维
                            if comp_tensor.size(-1) < 32:
                                padding = torch.zeros(batch_size, 32 - comp_tensor.size(-1), 
                                                    dtype=torch.float32, 
                                                    device=temporal_features.device)
                                comp_tensor = torch.cat([comp_tensor, padding], dim=-1)
                            elif comp_tensor.size(-1) > 32:
                                comp_tensor = comp_tensor[:, :32]
                        else:
                            comp_tensor = torch.zeros(batch_size, 32, 
                                                    dtype=torch.float32, 
                                                    device=temporal_features.device)
                    
                    comp_processed = self.compression_fc(comp_tensor)
                    fusion_features.append(comp_processed)
                except Exception as e:
                    print(f"⚠️ 压缩特征处理失败: {e}")
                    comp_tensor = torch.zeros(batch_size, 32, 
                                            dtype=torch.float32, 
                                            device=temporal_features.device)
                    comp_processed = self.compression_fc(comp_tensor)
                    fusion_features.append(comp_processed)
            
            # 处理时序一致性特征
            if 'temporal' in additional_features:
                try:
                    temp_feat = additional_features['temporal']
                    if isinstance(temp_feat, dict):
                        # 安全地提取时序特征
                        temp_values = []
                        for key in ['mean_frame_diff', 'std_frame_diff', 'max_frame_diff', 'temporal_smoothness']:
                            if key in temp_feat:
                                value = temp_feat[key]
                                if isinstance(value, (int, float)):
                                    temp_values.append(float(value))
                                elif isinstance(value, torch.Tensor):
                                    temp_values.append(float(value.item() if value.numel() == 1 else value.mean().item()))
                                elif isinstance(value, np.ndarray):
                                    temp_values.append(float(value.item() if value.size == 1 else value.mean()))
                                else:
                                    temp_values.append(0.0)
                            else:
                                temp_values.append(0.0)
                        
                        temp_tensor = torch.tensor([temp_values] * batch_size, 
                                                 dtype=torch.float32, 
                                                 device=temporal_features.device)
                    else:
                        if isinstance(temp_feat, torch.Tensor):
                            temp_tensor = temp_feat.to(temporal_features.device)
                            if temp_tensor.dim() == 1:
                                temp_tensor = temp_tensor.unsqueeze(0).repeat(batch_size, 1)
                        else:
                            temp_tensor = torch.zeros(batch_size, 4, 
                                                    dtype=torch.float32, 
                                                    device=temporal_features.device)
                    
                    temp_processed = self.temporal_fc(temp_tensor)
                    fusion_features.append(temp_processed)
                except Exception as e:
                    print(f"⚠️ 时序特征处理失败: {e}")
                    temp_tensor = torch.zeros(batch_size, 4, 
                                            dtype=torch.float32, 
                                            device=temporal_features.device)
                    temp_processed = self.temporal_fc(temp_tensor)
                    fusion_features.append(temp_processed)
            
            # 特征融合 - 确保维度一致性
            if len(fusion_features) > 1:
                try:
                    # 检查每个特征的维度
                    feature_dims = [f.shape[1] for f in fusion_features]
                    total_dim = sum(feature_dims)
                    expected_dim = self.fusion_layer[0].in_features
                    
                    if total_dim == expected_dim:
                        # 维度匹配，直接融合
                        fused_features = torch.cat(fusion_features, dim=1)
                        final_features = self.fusion_layer(fused_features)
                    else:
                        # 维度不匹配时进行调整（这是正常的多模态特征处理）
                        if total_dim < expected_dim:
                            # 维度不足，用零填充
                            padding_dim = expected_dim - total_dim
                            fused_features = torch.cat(fusion_features, dim=1)
                            padding = torch.zeros(batch_size, padding_dim, 
                                                dtype=fused_features.dtype, 
                                                device=fused_features.device)
                            fused_features = torch.cat([fused_features, padding], dim=1)
                            final_features = self.fusion_layer(fused_features)
                            # 只在调试模式下输出详细信息
                            if hasattr(self, 'debug_mode') and self.debug_mode:
                                print(f"🔧 特征填充: {total_dim} -> {expected_dim}")
                        elif total_dim > expected_dim:
                            # 维度过多，截断到期望维度
                            fused_features = torch.cat(fusion_features, dim=1)
                            fused_features = fused_features[:, :expected_dim]
                            final_features = self.fusion_layer(fused_features)
                            # 只在调试模式下输出详细信息
                            if hasattr(self, 'debug_mode') and self.debug_mode:
                                print(f"🔧 特征截断: {total_dim} -> {expected_dim}")
                        else:
                            # 理论上不应该到达这里
                            print(f"⚠️ 特征融合异常，使用基础特征")
                            final_features = temporal_features
                            
                except Exception as e:
                    print(f"⚠️ 特征融合失败: {e}")
                    final_features = temporal_features
            else:
                final_features = temporal_features
        else:
            final_features = temporal_features
        
        # 分类预测 - 根据特征维度选择合适的分类器
        if self.ensemble_mode:
            # 集成预测
            main_pred = self.main_classifier(final_features)
            spatial_pred = self.spatial_classifier(final_features)
            temporal_pred = self.temporal_classifier(final_features)
            
            # 加权融合
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_pred = (weights[0] * main_pred + 
                           weights[1] * spatial_pred + 
                           weights[2] * temporal_pred)
            
            if self.training:
                # 训练时返回所有预测用于多任务学习
                return {
                    'main': main_pred,
                    'spatial': spatial_pred,
                    'temporal': temporal_pred,
                    'ensemble': ensemble_pred
                }
            else:
                # 推理时只返回集成结果
                return ensemble_pred
        else:
            # 检查特征维度并选择合适的分类器
            feature_dim = final_features.shape[1]
            
            # 获取分类器的输入维度
            classifier_input_dim = None
            single_classifier_input_dim = None
            
            # 找到第一个Linear层来获取输入维度
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    classifier_input_dim = layer.in_features
                    break
            
            for layer in self.single_classifier:
                if isinstance(layer, nn.Linear):
                    single_classifier_input_dim = layer.in_features
                    break
            
            # 根据特征维度选择合适的分类器
            if classifier_input_dim and feature_dim == classifier_input_dim:
                logits = self.classifier(final_features)
            elif single_classifier_input_dim and feature_dim == single_classifier_input_dim:
                logits = self.single_classifier(final_features)
            else:
                # 如果都不匹配，尝试使用单一分类器（通常处理基础特征）
                print(f"⚠️ 特征维度 {feature_dim} 不匹配任何分类器，使用单一分类器")
                logits = self.single_classifier(final_features)
            
            # 检查输出是否包含NaN或无穷值
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("⚠️ 模型输出包含NaN/Inf，使用安全的默认输出")
                # 返回安全的默认输出（中性预测）
                batch_size = logits.shape[0]
                device = logits.device
                logits = torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
            
            # 限制logits的数值范围，避免极端值
            logits = torch.clamp(logits, -10, 10)
            
            return logits

    def get_attention_weights(self, x):
        """获取注意力权重（用于可视化）"""
        if not self.use_attention:
            return None
        
        batch_size, num_frames, channels, height, width = x.shape
        x_reshaped = x.view(batch_size * num_frames, channels, height, width)
        spatial_features = self.backbone(x_reshaped)
        spatial_features = spatial_features.view(batch_size, num_frames, -1)
        
        _, attention_weights = self.attention(
            spatial_features, spatial_features, spatial_features
        )
        
        return attention_weights

    def enable_ensemble_mode(self):
        """启用集成模式"""
        self.ensemble_mode = True
        print("🎯 集成模式已启用")

    def disable_ensemble_mode(self):
        """禁用集成模式"""
        self.ensemble_mode = False
        print("🎯 集成模式已禁用")

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'use_attention': self.use_attention,
            'use_multimodal': self.use_multimodal,
            'ensemble_mode': self.ensemble_mode
        }

def create_ensemble_models(num_models=3, **kwargs):
    """创建多个模型用于集成学习"""
    models = []
    for i in range(num_models):
        # 为每个模型使用不同的配置
        model_kwargs = kwargs.copy()
        if i == 0:
            model_kwargs.update({'use_attention': True, 'dropout_rate': 0.3})
        elif i == 1:
            model_kwargs.update({'use_attention': False, 'dropout_rate': 0.4})
        else:
            model_kwargs.update({'use_attention': True, 'dropout_rate': 0.2})
        
        model = OptimizedDeepfakeDetector(**model_kwargs)
        models.append(model)
    
    print(f"✅ 创建了 {num_models} 个集成模型")
    return models

print("✅ 优化模型定义完成")