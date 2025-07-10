# 视频处理优化模块 - 统一的视频处理解决方案

import torch
import numpy as np
import cv2
import torch.nn.functional as F
from typing import List, Tuple, Optional
import warnings
from memory_manager import get_memory_manager

# 检查PyAV可用性
try:
    import av
    from torchvision.io import read_video
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

class OptimizedVideoProcessor:
    """统一的视频处理器 - 支持GPU加速、内存优化和质量检测"""
    
    def __init__(self, 
                 max_memory_per_video: float = 1.0,  # 每个视频最大内存使用(GB)
                 chunk_size: int = 8,  # 分块处理大小
                 fallback_to_cpu: bool = True,
                 max_frames: int = 16,  # 最大帧数
                 target_size: Tuple[int, int] = (224, 224),  # 目标尺寸
                 quality_threshold: float = 20.0):  # 质量阈值
        
        self.max_memory_per_video = max_memory_per_video
        self.chunk_size = chunk_size
        self.fallback_to_cpu = fallback_to_cpu
        self.max_frames = max_frames
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_manager = get_memory_manager()
        
        # 预计算标准化参数
        if torch.cuda.is_available():
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def estimate_memory_usage(self, frames: List[np.ndarray]) -> float:
        """估算内存使用量"""
        if not frames:
            return 0.0
        
        # 估算单帧内存使用
        frame_shape = frames[0].shape
        bytes_per_frame = np.prod(frame_shape) * 4  # float32
        total_bytes = bytes_per_frame * len(frames)
        
        # 考虑GPU处理时的额外开销（约2倍）
        if self.device.type == 'cuda':
            total_bytes *= 2
        
        return total_bytes / (1024**3)  # 转换为GB
    
    def process_video_safe(self, frames: List[np.ndarray]) -> torch.Tensor:
        """安全的视频处理 - 自动处理内存不足问题"""
        if not frames:
            raise ValueError("帧列表为空")
        
        # 估算内存使用
        estimated_memory = self.estimate_memory_usage(frames)
        
        # 如果估算内存超过限制，使用分块处理
        if estimated_memory > self.max_memory_per_video:
            return self._process_video_chunked(frames)
        
        # 尝试GPU处理
        if self.device.type == 'cuda':
            try:
                return self._process_video_gpu(frames)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower():
                    # GPU内存不足，清理后重试
                    self.memory_manager.cleanup_gpu_memory(force=True)
                    try:
                        return self._process_video_gpu(frames)
                    except:
                        # 重试失败，回退到CPU或分块处理
                        if self.fallback_to_cpu:
                            warnings.warn("GPU内存不足，回退到CPU处理", UserWarning)
                            return self._process_video_cpu(frames)
                        else:
                            return self._process_video_chunked(frames)
                else:
                    raise
        else:
            return self._process_video_cpu(frames)
    
    def _process_video_gpu(self, frames: List[np.ndarray]) -> torch.Tensor:
        """GPU视频处理"""
        # 转换为tensor
        frames_array = np.stack(frames)  # (T, H, W, C)
        video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        
        # 移动到GPU并进行预处理
        video_tensor = video_tensor.to(self.device, non_blocking=True, dtype=torch.float32) / 255.0
        
        # 标准化
        video_tensor = (video_tensor - self.mean) / self.std
        
        return video_tensor
    
    def _process_video_cpu(self, frames: List[np.ndarray]) -> torch.Tensor:
        """CPU视频处理"""
        frames_tensors = []
        for frame in frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames_tensors.append(frame_tensor)
        
        video_tensor = torch.stack(frames_tensors)
        
        # 确保标准化参数在CPU上
        mean_cpu = self.mean.cpu() if self.mean.is_cuda else self.mean
        std_cpu = self.std.cpu() if self.std.is_cuda else self.std
        
        # 标准化
        video_tensor = (video_tensor - mean_cpu) / std_cpu
        
        return video_tensor
    
    def _process_video_chunked(self, frames: List[np.ndarray]) -> torch.Tensor:
        """分块处理视频"""
        processed_chunks = []
        
        for i in range(0, len(frames), self.chunk_size):
            chunk_frames = frames[i:i + self.chunk_size]
            
            # 处理当前块
            if self.device.type == 'cuda':
                try:
                    chunk_tensor = self._process_video_gpu(chunk_frames)
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    # 块处理也失败，回退到CPU
                    chunk_tensor = self._process_video_cpu(chunk_frames)
            else:
                chunk_tensor = self._process_video_cpu(chunk_frames)
            
            processed_chunks.append(chunk_tensor)
            
            # 定期清理内存
            if i % (self.chunk_size * 4) == 0:
                self.memory_manager.smart_cleanup()
        
        # 合并所有块
        return torch.cat(processed_chunks, dim=0)
    
    def read_video_safe(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """安全的视频读取 - 处理内存分配失败"""
        frames = []
        
        try:
            # 尝试使用OpenCV读取
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换颜色空间
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
                
                # 检查内存使用，防止过度消耗
                if frame_count % 10 == 0:
                    estimated_memory = self.estimate_memory_usage(frames)
                    if estimated_memory > self.max_memory_per_video * 2:  # 预留缓冲
                        warnings.warn(f"视频内存使用过高，停止读取更多帧: {estimated_memory:.2f}GB", UserWarning)
                        break
            
            cap.release()
            
        except Exception as e:
            warnings.warn(f"视频读取失败: {e}", UserWarning)
            return []
        
        return frames
    
    def extract_frames_gpu_accelerated(self, video_path: str, use_gpu: bool = True) -> List[np.ndarray]:
        """GPU加速的帧提取函数 - 兼容data_processing.py接口"""
        try:
            # 检查PyAV是否可用
            if not PYAV_AVAILABLE:
                warnings.warn(f"PyAV不可用，使用CPU回退处理: {video_path}", UserWarning)
                return self._extract_frames_cpu_fallback(video_path)
                
            # 使用torchvision的GPU加速视频读取
            if use_gpu and torch.cuda.is_available():
                device = self.device
            else:
                device = torch.device('cpu')
                
            # 读取视频（torchvision自动处理解码）
            try:
                video_tensor, audio, info = read_video(video_path, pts_unit='sec')
                # video_tensor shape: (T, H, W, C)
            except Exception as e:
                warnings.warn(f"GPU视频读取失败，回退到CPU: {e}", UserWarning)
                return self._extract_frames_cpu_fallback(video_path)
            
            if video_tensor.size(0) == 0:
                return []
                
            # 移动到GPU进行处理
            video_tensor = video_tensor.to(device, non_blocking=True)
            total_frames = video_tensor.size(0)
            
            # 智能帧采样策略
            if total_frames <= self.max_frames:
                frame_indices = torch.arange(0, total_frames, device=device)
            else:
                # 均匀采样
                step = total_frames / self.max_frames
                frame_indices = torch.arange(0, total_frames, step, device=device).long()[:self.max_frames]
            
            # 批量提取帧
            selected_frames = video_tensor[frame_indices]  # (max_frames, H, W, C)
            
            # GPU上进行质量检测（使用Sobel算子代替Laplacian）
            if self.quality_threshold > 0:
                # 转换为灰度图进行质量检测（先转换为float类型）
                gray_frames = selected_frames.float().mean(dim=-1, keepdim=True)  # (T, H, W, 1)
                gray_frames = gray_frames.permute(0, 3, 1, 2)  # (T, 1, H, W)
                
                # 使用Sobel算子计算图像质量
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32, device=device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32, device=device).view(1, 1, 3, 3)
                
                grad_x = F.conv2d(gray_frames, sobel_x, padding=1)
                grad_y = F.conv2d(gray_frames, sobel_y, padding=1)
                quality_scores = (grad_x.pow(2) + grad_y.pow(2)).mean(dim=[1, 2, 3])
                
                # 过滤低质量帧
                quality_mask = quality_scores > self.quality_threshold
                if quality_mask.sum() > 0:
                    selected_frames = selected_frames[quality_mask]
                
            # GPU上进行尺寸调整
            selected_frames = selected_frames.permute(0, 3, 1, 2).float()  # (T, C, H, W)
            if selected_frames.size(-1) != self.target_size[0] or selected_frames.size(-2) != self.target_size[1]:
                selected_frames = F.interpolate(selected_frames, size=self.target_size, 
                                              mode='bilinear', align_corners=False)
            
            # 确保帧数足够
            current_frames = selected_frames.size(0)
            if current_frames < self.max_frames:
                # 重复最后一帧
                if current_frames > 0:
                    last_frame = selected_frames[-1:].repeat(self.max_frames - current_frames, 1, 1, 1)
                    selected_frames = torch.cat([selected_frames, last_frame], dim=0)
                else:
                    # 创建黑色帧
                    selected_frames = torch.zeros(self.max_frames, 3, self.target_size[0], self.target_size[1], 
                                                device=device, dtype=torch.float32)
            
            # 限制到最大帧数
            selected_frames = selected_frames[:self.max_frames]
            
            # 转换回CPU numpy格式（为了兼容现有代码）
            frames_cpu = selected_frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            frames_list = [frame for frame in frames_cpu]
            
            return frames_list
            
        except Exception as e:
            warnings.warn(f"GPU帧提取失败，回退到CPU: {e}", UserWarning)
            return self._extract_frames_cpu_fallback(video_path)
    
    def _extract_frames_cpu_fallback(self, video_path: str) -> List[np.ndarray]:
        """CPU回退的帧提取函数"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            warnings.warn(f"无法打开视频: {video_path}", UserWarning)
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return frames

        # 均匀采样策略
        if total_frames <= self.max_frames:
            frame_indices = list(range(0, total_frames, max(1, total_frames // self.max_frames)))
        else:
            step = max(1, total_frames // self.max_frames)
            frame_indices = list(range(0, total_frames, step))[:self.max_frames]

        frame_count = 0
        for frame_idx in frame_indices:
            if frame_count >= self.max_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 质量检测
                if self.quality_threshold > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    quality = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if quality <= self.quality_threshold:
                        continue
                
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
                frame_count += 1

        cap.release()

        # 如果帧数不足，重复最后一帧
        while len(frames) < self.max_frames and len(frames) > 0:
            frames.append(frames[-1].copy())

        return frames[:self.max_frames]

# 全局视频处理器实例
_global_video_processor: Optional[OptimizedVideoProcessor] = None

def get_video_processor(max_frames: int = None, target_size: Tuple[int, int] = None, 
                       quality_threshold: float = None) -> OptimizedVideoProcessor:
    """获取视频处理器实例
    
    Args:
        max_frames: 最大帧数，如果为None则使用默认值
        target_size: 目标尺寸，如果为None则使用默认值
        quality_threshold: 质量阈值，如果为None则使用默认值
    
    Returns:
        OptimizedVideoProcessor实例
    """
    global _global_video_processor
    
    # 如果没有提供参数，返回全局实例
    if max_frames is None and target_size is None and quality_threshold is None:
        if _global_video_processor is None:
            _global_video_processor = OptimizedVideoProcessor()
        return _global_video_processor
    
    # 如果提供了参数，创建新的实例
    kwargs = {}
    if max_frames is not None:
        kwargs['max_frames'] = max_frames
    if target_size is not None:
        kwargs['target_size'] = target_size
    if quality_threshold is not None:
        kwargs['quality_threshold'] = quality_threshold
    
    return OptimizedVideoProcessor(**kwargs)

def process_video_safe(frames: List[np.ndarray]) -> torch.Tensor:
    """快速视频处理函数"""
    processor = get_video_processor()
    return processor.process_video_safe(frames)

def read_video_safe(video_path: str, max_frames: int = None) -> List[np.ndarray]:
    """快速视频读取函数"""
    processor = get_video_processor()
    return processor.read_video_safe(video_path, max_frames)