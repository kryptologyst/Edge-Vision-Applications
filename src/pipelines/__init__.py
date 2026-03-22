"""Data processing and inference pipelines for Edge Vision Applications."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..utils import PerformanceProfiler, validate_input_shape, mask_sensitive_data


logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing pipeline for edge vision applications."""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augment: bool = False):
        """Initialize image preprocessor.
        
        Args:
            input_size: Target input size (height, width)
            normalize: Whether to normalize pixel values
            augment: Whether to apply data augmentation
        """
        self.input_size = input_size
        self.normalize = normalize
        self.augment = augment
        
        # Define transforms
        self._setup_transforms()
        
    def _setup_transforms(self) -> None:
        """Setup image transformation pipeline."""
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize(self.input_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if requested
        if self.normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
            
        # Data augmentation if requested
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
            
        self.transform = transforms.Compose(transform_list)
        
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
        return self.transform(image)


class CameraCapture:
    """Camera capture utility for real-time video processing."""
    
    def __init__(self, 
                 camera_id: int = 0,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30):
        """Initialize camera capture.
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
            fps: Target FPS
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        
    def __enter__(self):
        """Context manager entry."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.cap:
            self.cap.release()
            
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the camera.
        
        Returns:
            Camera frame or None if failed
        """
        if not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame


class InferencePipeline:
    """Main inference pipeline for edge vision applications."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 preprocessor: ImagePreprocessor,
                 device: str = "cpu",
                 batch_size: int = 1):
        """Initialize inference pipeline.
        
        Args:
            model: Trained model for inference
            preprocessor: Image preprocessor
            device: Device for inference
            batch_size: Batch size for inference
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.batch_size = batch_size
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize performance tracking
        self.inference_times = []
        self.frame_count = 0
        
    def predict_single(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """Run inference on a single image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        with PerformanceProfiler("single_inference") as profiler:
            # Preprocess image
            input_tensor = self.preprocessor.preprocess(image)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Validate input shape
            expected_shape = (1, 3, *self.preprocessor.input_size)
            if not validate_input_shape(input_tensor, expected_shape):
                logger.warning(f"Unexpected input shape: {input_tensor.shape}")
                
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
            
            # Convert to CPU and numpy
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            # Format results
            results = {
                "predictions": [
                    {"class_id": int(idx), "probability": float(prob)}
                    for idx, prob in zip(top_indices, top_probs)
                ],
                "top_prediction": {
                    "class_id": int(top_indices[0]),
                    "probability": float(top_probs[0])
                }
            }
            
            return results
            
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict]:
        """Run inference on a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process images in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocessor.preprocess(img)
                batch_tensors.append(tensor)
                
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
            # Process results
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
            
            for j in range(len(batch_images)):
                batch_results = {
                    "predictions": [
                        {"class_id": int(idx), "probability": float(prob)}
                        for idx, prob in zip(top_indices[j], top_probs[j])
                    ],
                    "top_prediction": {
                        "class_id": int(top_indices[j][0]),
                        "probability": float(top_probs[j][0])
                    }
                }
                results.append(batch_results)
                
        return results
        
    def process_video_stream(self, 
                           camera: CameraCapture,
                           max_frames: Optional[int] = None,
                           privacy_mask: bool = False) -> List[Dict]:
        """Process video stream from camera.
        
        Args:
            camera: Camera capture instance
            max_frames: Maximum number of frames to process
            privacy_mask: Whether to apply privacy masking
            
        Returns:
            List of frame processing results
        """
        results = []
        frame_count = 0
        
        with camera:
            while True:
                if max_frames and frame_count >= max_frames:
                    break
                    
                frame = camera.read_frame()
                if frame is None:
                    break
                    
                # Apply privacy masking if requested
                if privacy_mask:
                    # Example: mask face regions (simplified)
                    mask_regions = [(100, 100, 200, 200)]  # x, y, w, h
                    frame = mask_sensitive_data(frame, mask_regions)
                    
                # Process frame
                result = self.predict_single(frame)
                result["frame_id"] = frame_count
                result["timestamp"] = frame_count / camera.fps
                
                results.append(result)
                frame_count += 1
                
                # Log progress
                if frame_count % 30 == 0:  # Every second at 30 FPS
                    logger.info(f"Processed {frame_count} frames")
                    
        return results
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {}
            
        times = np.array(self.inference_times)
        
        return {
            "mean_latency_ms": float(np.mean(times)),
            "std_latency_ms": float(np.std(times)),
            "p50_latency_ms": float(np.percentile(times, 50)),
            "p95_latency_ms": float(np.percentile(times, 95)),
            "p99_latency_ms": float(np.percentile(times, 99)),
            "total_frames": self.frame_count,
            "avg_fps": 1000.0 / np.mean(times) if np.mean(times) > 0 else 0.0
        }


class EdgeInferenceOptimizer:
    """Optimizer for edge inference performance."""
    
    def __init__(self, model: torch.nn.Module):
        """Initialize edge inference optimizer.
        
        Args:
            model: Model to optimize
        """
        self.model = model
        
    def optimize_for_edge(self, 
                         target_device: str = "cpu",
                         optimization_level: str = "high") -> torch.nn.Module:
        """Optimize model for edge deployment.
        
        Args:
            target_device: Target device for optimization
            optimization_level: Level of optimization ('low', 'medium', 'high')
            
        Returns:
            Optimized model
        """
        logger.info(f"Optimizing model for {target_device} with {optimization_level} optimization")
        
        # Apply optimizations based on level
        if optimization_level == "high":
            self.model = self._apply_high_optimization()
        elif optimization_level == "medium":
            self.model = self._apply_medium_optimization()
        else:
            self.model = self._apply_low_optimization()
            
        return self.model
        
    def _apply_low_optimization(self) -> torch.nn.Module:
        """Apply low-level optimizations."""
        # Enable inference mode
        self.model.eval()
        return self.model
        
    def _apply_medium_optimization(self) -> torch.nn.Module:
        """Apply medium-level optimizations."""
        self.model.eval()
        
        # Enable optimizations for inference
        torch.backends.cudnn.benchmark = True
        
        return self.model
        
    def _apply_high_optimization(self) -> torch.nn.Module:
        """Apply high-level optimizations."""
        self.model.eval()
        
        # Enable all optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Compile model if PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
                
        return self.model
