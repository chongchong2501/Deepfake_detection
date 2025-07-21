# Cell 15: 集成推理脚本 - 整合所有优化功能（训练完成后使用）

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EnsembleDeepfakeDetector:
    """集成深度伪造检测器 - 整合所有优化功能"""
    
    def __init__(self, model_paths=None, device=None, use_mtcnn=True, 
                 extract_fourier=True, extract_compression=True):
        """
        初始化集成检测器
        
        Args:
            model_paths: 模型文件路径列表
            device: 计算设备
            use_mtcnn: 是否使用MTCNN人脸检测
            extract_fourier: 是否提取频域特征
            extract_compression: 是否提取压缩伪影特征
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mtcnn = use_mtcnn and globals().get('MTCNN_AVAILABLE', False)
        self.extract_fourier = extract_fourier and globals().get('SCIPY_AVAILABLE', False)
        self.extract_compression = extract_compression
        
        # 初始化MTCNN
        if self.use_mtcnn:
            try:
                from facenet_pytorch import MTCNN
                self.mtcnn = MTCNN(
                    image_size=224,
                    margin=20,
                    min_face_size=50,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=True,
                    device=self.device
                )
                print("✅ MTCNN人脸检测器初始化成功")
            except Exception as e:
                print(f"⚠️ MTCNN初始化失败: {e}")
                self.use_mtcnn = False
                self.mtcnn = None
        else:
            self.mtcnn = None
        
        # 加载模型
        self.models = []
        if model_paths:
            self.load_models(model_paths)
        
        print(f"🚀 集成检测器初始化完成")
        print(f"   - 设备: {self.device}")
        print(f"   - MTCNN: {'启用' if self.use_mtcnn else '禁用'}")
        print(f"   - 频域分析: {'启用' if self.extract_fourier else '禁用'}")
        print(f"   - 压缩分析: {'启用' if self.extract_compression else '禁用'}")

    def load_models(self, model_paths):
        """加载多个模型用于集成"""
        from .cell_05_model_definition import OptimizedDeepfakeDetector
        
        for i, model_path in enumerate(model_paths):
            try:
                # 创建模型实例
                model = OptimizedDeepfakeDetector(
                    use_attention=True,
                    use_multimodal=True,
                    ensemble_mode=True
                )
                
                # 加载权重
                if Path(model_path).exists():
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
                    print(f"✅ 模型 {i+1} 加载成功: {model_path}")
                else:
                    print(f"⚠️ 模型文件不存在: {model_path}")
                    
            except Exception as e:
                print(f"⚠️ 加载模型 {model_path} 失败: {e}")
        
        print(f"📊 成功加载 {len(self.models)} 个模型用于集成")

    def extract_frames_from_video(self, video_path, max_frames=16, target_size=(224, 224)):
        """从视频中提取帧"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 计算采样间隔
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 转换颜色空间
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # MTCNN人脸检测和裁剪
                    if self.use_mtcnn and self.mtcnn is not None:
                        try:
                            # 检测人脸
                            face = self.mtcnn(frame)
                            if face is not None:
                                # 转换为numpy数组并调整大小
                                face_np = face.permute(1, 2, 0).cpu().numpy()
                                face_np = (face_np * 255).astype(np.uint8)
                                frame = cv2.resize(face_np, target_size)
                            else:
                                # 如果没有检测到人脸，使用原始帧
                                frame = cv2.resize(frame, target_size)
                        except Exception as e:
                            print(f"⚠️ MTCNN处理失败: {e}")
                            frame = cv2.resize(frame, target_size)
                    else:
                        # 直接调整大小
                        frame = cv2.resize(frame, target_size)
                    
                    frames.append(frame)
            
            cap.release()
            
            # 确保帧数一致
            while len(frames) < max_frames:
                frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))
            
            return frames[:max_frames]
            
        except Exception as e:
            print(f"⚠️ 提取视频帧失败: {e}")
            # 返回默认帧
            return [np.zeros((*target_size, 3), dtype=np.uint8) for _ in range(max_frames)]

    def extract_additional_features(self, frames):
        """提取额外的多模态特征"""
        features = {}
        
        try:
            # 频域特征提取
            if self.extract_fourier and len(frames) > 0:
                mid_frame = frames[len(frames) // 2]
                from .cell_03_data_processing import extract_fourier_features
                fourier_features = extract_fourier_features(mid_frame)
                if fourier_features:
                    features['fourier'] = fourier_features
            
            # 压缩伪影特征提取
            if self.extract_compression and len(frames) > 0:
                compression_features = []
                for frame in frames[::4]:  # 每4帧采样一次
                    from .cell_03_data_processing import analyze_compression_artifacts
                    comp_feat = analyze_compression_artifacts(frame)
                    if comp_feat:
                        compression_features.append(comp_feat)
                
                if compression_features:
                    features['compression'] = {
                        'mean_dct_energy': np.mean([f['dct_energy'] for f in compression_features]),
                        'mean_edge_density': np.mean([f['edge_density'] for f in compression_features]),
                        'std_dct_energy': np.std([f['dct_energy'] for f in compression_features])
                    }
            
            # 时序一致性特征
            if len(frames) > 1:
                frame_diffs = []
                for i in range(len(frames) - 1):
                    diff = np.mean(np.abs(frames[i+1].astype(float) - frames[i].astype(float)))
                    frame_diffs.append(diff)
                
                if frame_diffs:
                    features['temporal'] = {
                        'mean_frame_diff': np.mean(frame_diffs),
                        'std_frame_diff': np.std(frame_diffs),
                        'max_frame_diff': np.max(frame_diffs),
                        'temporal_smoothness': 1.0 / (1.0 + np.std(frame_diffs))
                    }
            
            return features if features else None
            
        except Exception as e:
            print(f"⚠️ 提取额外特征失败: {e}")
            return None

    def preprocess_frames(self, frames):
        """预处理帧数据"""
        try:
            # 转换为张量
            video_tensor = torch.stack([
                torch.from_numpy(frame).permute(2, 0, 1) for frame in frames
            ]).float() / 255.0  # (T, C, H, W)
            
            # 标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            video_tensor = (video_tensor - mean) / std
            
            # 添加批次维度
            video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
            
            return video_tensor.to(self.device)
            
        except Exception as e:
            print(f"⚠️ 预处理失败: {e}")
            return None

    def predict_single_video(self, video_path, return_details=False):
        """对单个视频进行预测"""
        try:
            # 提取帧
            frames = self.extract_frames_from_video(video_path)
            if not frames:
                return {'error': '无法提取视频帧'}
            
            # 提取额外特征
            additional_features = self.extract_additional_features(frames)
            
            # 预处理
            video_tensor = self.preprocess_frames(frames)
            if video_tensor is None:
                return {'error': '预处理失败'}
            
            # 集成预测
            predictions = []
            model_outputs = []
            
            with torch.no_grad():
                for i, model in enumerate(self.models):
                    try:
                        if additional_features:
                            # 处理额外特征为张量格式
                            processed_features = {}
                            for key, value in additional_features.items():
                                if isinstance(value, dict):
                                    # 将字典转换为张量
                                    if key == 'fourier':
                                        tensor_value = torch.tensor(
                                            list(value.values()), dtype=torch.float32
                                        ).unsqueeze(0).to(self.device)
                                    elif key == 'compression':
                                        tensor_value = torch.tensor([
                                            value['mean_dct_energy'],
                                            value['mean_edge_density'],
                                            value['std_dct_energy']
                                        ], dtype=torch.float32).unsqueeze(0).to(self.device)
                                    elif key == 'temporal':
                                        tensor_value = torch.tensor([
                                            value['mean_frame_diff'],
                                            value['std_frame_diff'],
                                            value['max_frame_diff'],
                                            value['temporal_smoothness']
                                        ], dtype=torch.float32).unsqueeze(0).to(self.device)
                                    else:
                                        tensor_value = torch.tensor(value).to(self.device)
                                    
                                    processed_features[key] = tensor_value
                            
                            outputs = model(video_tensor, processed_features)
                        else:
                            outputs = model(video_tensor)
                        
                        # 处理输出
                        if isinstance(outputs, dict):
                            # 集成模式，使用ensemble输出
                            pred = outputs['ensemble']
                            model_outputs.append({
                                'main': torch.sigmoid(outputs['main']).item(),
                                'spatial': torch.sigmoid(outputs['spatial']).item(),
                                'temporal': torch.sigmoid(outputs['temporal']).item(),
                                'ensemble': torch.sigmoid(outputs['ensemble']).item()
                            })
                        else:
                            pred = outputs
                            model_outputs.append({'prediction': torch.sigmoid(pred).item()})
                        
                        if pred.dim() > 1:
                            pred = pred.squeeze(-1)
                        
                        pred_prob = torch.sigmoid(pred).item()
                        predictions.append(pred_prob)
                        
                    except Exception as e:
                        print(f"⚠️ 模型 {i+1} 预测失败: {e}")
                        continue
            
            if not predictions:
                return {'error': '所有模型预测失败'}
            
            # 计算集成结果
            ensemble_prob = np.mean(predictions)
            ensemble_pred = 'FAKE' if ensemble_prob > 0.5 else 'REAL'
            confidence = max(ensemble_prob, 1 - ensemble_prob)
            
            result = {
                'prediction': ensemble_pred,
                'probability': ensemble_prob,
                'confidence': confidence,
                'individual_predictions': predictions,
                'num_models': len(predictions)
            }
            
            if return_details:
                result.update({
                    'model_outputs': model_outputs,
                    'additional_features': additional_features,
                    'num_frames': len(frames)
                })
            
            return result
            
        except Exception as e:
            return {'error': f'预测失败: {str(e)}'}

    def predict_batch(self, video_paths, show_progress=True):
        """批量预测多个视频"""
        results = {}
        
        iterator = tqdm(video_paths, desc="批量预测中") if show_progress else video_paths
        
        for video_path in iterator:
            try:
                result = self.predict_single_video(video_path)
                results[str(video_path)] = result
            except Exception as e:
                results[str(video_path)] = {'error': f'处理失败: {str(e)}'}
        
        return results

    def visualize_prediction(self, video_path, save_path=None):
        """可视化预测结果"""
        try:
            # 获取预测结果
            result = self.predict_single_video(video_path, return_details=True)
            
            if 'error' in result:
                print(f"❌ 预测失败: {result['error']}")
                return
            
            # 提取帧用于可视化
            frames = self.extract_frames_from_video(video_path, max_frames=8)
            
            # 创建可视化
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f'深度伪造检测结果\n预测: {result["prediction"]} (置信度: {result["confidence"]:.3f})', 
                        fontsize=16, fontweight='bold')
            
            # 显示帧
            for i, frame in enumerate(frames):
                row = i // 4
                col = i % 4
                axes[row, col].imshow(frame)
                axes[row, col].set_title(f'帧 {i+1}')
                axes[row, col].axis('off')
            
            # 添加预测信息
            info_text = f"""
            视频路径: {Path(video_path).name}
            预测结果: {result['prediction']}
            概率: {result['probability']:.4f}
            置信度: {result['confidence']:.4f}
            使用模型数: {result['num_models']}
            提取帧数: {result.get('num_frames', 'N/A')}
            """
            
            fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 可视化结果已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")

    def get_model_info(self):
        """获取模型信息"""
        info = {
            'num_models': len(self.models),
            'device': str(self.device),
            'use_mtcnn': self.use_mtcnn,
            'extract_fourier': self.extract_fourier,
            'extract_compression': self.extract_compression,
            'model_details': []
        }
        
        for i, model in enumerate(self.models):
            try:
                model_info = model.get_model_info()
                info['model_details'].append({
                    'model_id': i+1,
                    **model_info
                })
            except:
                info['model_details'].append({
                    'model_id': i+1,
                    'error': '无法获取模型信息'
                })
        
        return info

# 便捷函数
def create_ensemble_detector(model_paths=None, **kwargs):
    """创建集成检测器的便捷函数"""
    return EnsembleDeepfakeDetector(model_paths=model_paths, **kwargs)

def quick_predict(video_path, model_paths=None):
    """快速预测单个视频"""
    detector = create_ensemble_detector(model_paths)
    return detector.predict_single_video(video_path)

print("✅ 集成推理脚本定义完成")