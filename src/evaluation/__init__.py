"""Evaluation and metrics for Edge Vision Applications."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import PerformanceProfiler, get_device


logger = logging.getLogger(__name__)


class AccuracyEvaluator:
    """Evaluate model accuracy on test datasets."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = "cpu",
                 num_classes: int = 1000):
        """Initialize accuracy evaluator.
        
        Args:
            model: Model to evaluate
            device: Device for evaluation
            num_classes: Number of classes
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
    def evaluate_dataset(self, 
                        dataloader: torch.utils.data.DataLoader,
                        top_k: List[int] = [1, 5]) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            top_k: List of k values for top-k accuracy
            
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Starting dataset evaluation")
        
        correct = {k: 0 for k in top_k}
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate top-k accuracy
                _, pred = output.topk(max(top_k), 1, largest=True, sorted=True)
                pred = pred.t()
                correct_batch = pred.eq(target.view(1, -1).expand_as(pred))
                
                for k in top_k:
                    correct[k] += correct_batch[:k].reshape(-1).float().sum().item()
                    
                total += target.size(0)
                
                if batch_idx % 100 == 0:
                    logger.info(f"Evaluated {batch_idx * len(data)} samples")
                    
        # Calculate final accuracies
        accuracies = {}
        for k in top_k:
            accuracies[f"top_{k}_accuracy"] = correct[k] / total
            
        logger.info(f"Evaluation completed. Top-1: {accuracies['top_1_accuracy']:.4f}")
        return accuracies
        
    def evaluate_single_batch(self, 
                             data: torch.Tensor,
                             target: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on a single batch.
        
        Args:
            data: Input data tensor
            target: Target labels tensor
            
        Returns:
            Dictionary with accuracy metrics
        """
        data, target = data.to(self.device), target.to(self.device)
        
        with torch.no_grad():
            output = self.model(data)
            _, predicted = torch.max(output.data, 1)
            
        accuracy = (predicted == target).float().mean().item()
        
        return {
            "accuracy": accuracy,
            "correct": (predicted == target).sum().item(),
            "total": target.size(0)
        }


class PerformanceProfiler:
    """Profile model performance metrics."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """Initialize performance profiler.
        
        Args:
            model: Model to profile
            device: Device for profiling
        """
        self.model = model
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
    def profile_latency(self, 
                       input_shape: Tuple[int, ...],
                       num_runs: int = 100,
                       warmup_runs: int = 10) -> Dict[str, float]:
        """Profile model latency.
        
        Args:
            input_shape: Input tensor shape
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with latency metrics
        """
        logger.info(f"Profiling latency with {num_runs} runs")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)
                
        # Profile runs
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(dummy_input)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
                
        # Calculate statistics
        times = np.array(times)
        
        metrics = {
            "mean_latency_ms": float(np.mean(times)),
            "std_latency_ms": float(np.std(times)),
            "p50_latency_ms": float(np.percentile(times, 50)),
            "p95_latency_ms": float(np.percentile(times, 95)),
            "p99_latency_ms": float(np.percentile(times, 99)),
            "min_latency_ms": float(np.min(times)),
            "max_latency_ms": float(np.max(times)),
        }
        
        logger.info(f"Latency profiling completed. Mean: {metrics['mean_latency_ms']:.2f}ms")
        return metrics
        
    def profile_throughput(self, 
                          input_shape: Tuple[int, ...],
                          duration_seconds: int = 10) -> Dict[str, float]:
        """Profile model throughput.
        
        Args:
            input_shape: Input tensor shape
            duration_seconds: Duration of profiling
            
        Returns:
            Dictionary with throughput metrics
        """
        logger.info(f"Profiling throughput for {duration_seconds} seconds")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
                
        # Profile throughput
        start_time = time.time()
        inference_count = 0
        
        with torch.no_grad():
            while time.time() - start_time < duration_seconds:
                _ = self.model(dummy_input)
                inference_count += 1
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    
        end_time = time.time()
        total_time = end_time - start_time
        
        metrics = {
            "total_inferences": inference_count,
            "total_time_seconds": total_time,
            "inferences_per_second": inference_count / total_time,
            "fps": inference_count / total_time,
        }
        
        logger.info(f"Throughput profiling completed. FPS: {metrics['fps']:.2f}")
        return metrics
        
    def profile_memory_usage(self, 
                           input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """Profile model memory usage.
        
        Args:
            input_shape: Input tensor shape
            
        Returns:
            Dictionary with memory metrics
        """
        logger.info("Profiling memory usage")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # Measure memory before inference
        if self.device == "cuda":
            memory_before = torch.cuda.memory_allocated()
        else:
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
        # Run inference
        with torch.no_grad():
            _ = self.model(dummy_input)
            
        # Measure memory after inference
        if self.device == "cuda":
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
            peak_memory_mb = peak_memory / 1024 / 1024  # MB
        else:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            peak_memory_mb = memory_used
            
        metrics = {
            "memory_used_mb": memory_used,
            "peak_memory_mb": peak_memory_mb,
            "memory_before_mb": memory_before / 1024 / 1024 if self.device == "cuda" else memory_before,
            "memory_after_mb": memory_after / 1024 / 1024 if self.device == "cuda" else memory_after,
        }
        
        logger.info(f"Memory profiling completed. Used: {memory_used:.2f}MB")
        return metrics


class RobustnessEvaluator:
    """Evaluate model robustness to various perturbations."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: str = "cpu"):
        """Initialize robustness evaluator.
        
        Args:
            model: Model to evaluate
            device: Device for evaluation
        """
        self.model = model
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
    def evaluate_noise_robustness(self, 
                                 dataloader: torch.utils.data.DataLoader,
                                 noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict[str, Dict[str, float]]:
        """Evaluate robustness to Gaussian noise.
        
        Args:
            dataloader: DataLoader for evaluation
            noise_levels: List of noise standard deviations
            
        Returns:
            Dictionary with robustness metrics for each noise level
        """
        logger.info("Evaluating noise robustness")
        
        results = {}
        
        for noise_level in noise_levels:
            logger.info(f"Testing noise level: {noise_level}")
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Add Gaussian noise
                    noise = torch.randn_like(data) * noise_level
                    noisy_data = torch.clamp(data + noise, 0, 1)
                    
                    # Forward pass
                    output = self.model(noisy_data)
                    _, predicted = torch.max(output.data, 1)
                    
                    correct += (predicted == target).sum().item()
                    total += target.size(0)
                    
            accuracy = correct / total
            results[f"noise_{noise_level}"] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
            
        return results
        
    def evaluate_blur_robustness(self, 
                                dataloader: torch.utils.data.DataLoader,
                                blur_kernels: List[int] = [3, 5, 7, 9]) -> Dict[str, Dict[str, float]]:
        """Evaluate robustness to motion blur.
        
        Args:
            dataloader: DataLoader for evaluation
            blur_kernels: List of blur kernel sizes
            
        Returns:
            Dictionary with robustness metrics for each blur level
        """
        logger.info("Evaluating blur robustness")
        
        results = {}
        
        for kernel_size in blur_kernels:
            logger.info(f"Testing blur kernel size: {kernel_size}")
            
            # Create blur kernel
            blur_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
            blur_kernel = blur_kernel.to(self.device)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Apply blur
                    blurred_data = torch.nn.functional.conv2d(
                        data, blur_kernel, padding=kernel_size//2, groups=data.size(1)
                    )
                    
                    # Forward pass
                    output = self.model(blurred_data)
                    _, predicted = torch.max(output.data, 1)
                    
                    correct += (predicted == target).sum().item()
                    total += target.size(0)
                    
            accuracy = correct / total
            results[f"blur_{kernel_size}"] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
            
        return results


class ModelComparator:
    """Compare different model variants."""
    
    def __init__(self):
        """Initialize model comparator."""
        self.results = {}
        
    def add_model_results(self, 
                         model_name: str,
                         accuracy: float,
                         latency_ms: float,
                         model_size_mb: float,
                         memory_mb: float) -> None:
        """Add results for a model variant.
        
        Args:
            model_name: Name of the model variant
            accuracy: Model accuracy
            latency_ms: Inference latency in milliseconds
            model_size_mb: Model size in MB
            memory_mb: Memory usage in MB
        """
        self.results[model_name] = {
            "accuracy": accuracy,
            "latency_ms": latency_ms,
            "model_size_mb": model_size_mb,
            "memory_mb": memory_mb,
            "efficiency_score": accuracy / latency_ms,  # Accuracy per ms
            "size_efficiency": accuracy / model_size_mb,  # Accuracy per MB
        }
        
    def generate_comparison_report(self) -> str:
        """Generate a comparison report.
        
        Returns:
            Formatted comparison report
        """
        if not self.results:
            return "No results to compare"
            
        report = "Model Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        # Sort by efficiency score
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]["efficiency_score"],
            reverse=True
        )
        
        report += "Ranked by Efficiency Score (Accuracy/Latency):\n"
        report += "-" * 50 + "\n"
        
        for i, (model_name, metrics) in enumerate(sorted_models, 1):
            report += f"{i}. {model_name}:\n"
            report += f"   Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"   Latency: {metrics['latency_ms']:.2f} ms\n"
            report += f"   Model Size: {metrics['model_size_mb']:.2f} MB\n"
            report += f"   Memory: {metrics['memory_mb']:.2f} MB\n"
            report += f"   Efficiency Score: {metrics['efficiency_score']:.6f}\n"
            report += f"   Size Efficiency: {metrics['size_efficiency']:.6f}\n\n"
            
        return report
        
    def plot_comparison(self, 
                       save_path: Optional[str] = None) -> None:
        """Plot model comparison charts.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No results to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        model_names = list(self.results.keys())
        accuracies = [self.results[name]["accuracy"] for name in model_names]
        latencies = [self.results[name]["latency_ms"] for name in model_names]
        sizes = [self.results[name]["model_size_mb"] for name in model_names]
        memories = [self.results[name]["memory_mb"] for name in model_names]
        
        # Accuracy vs Latency
        axes[0, 0].scatter(latencies, accuracies, s=100, alpha=0.7)
        axes[0, 0].set_xlabel("Latency (ms)")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Accuracy vs Latency")
        for i, name in enumerate(model_names):
            axes[0, 0].annotate(name, (latencies[i], accuracies[i]))
            
        # Accuracy vs Model Size
        axes[0, 1].scatter(sizes, accuracies, s=100, alpha=0.7)
        axes[0, 1].set_xlabel("Model Size (MB)")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Accuracy vs Model Size")
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name, (sizes[i], accuracies[i]))
            
        # Latency vs Model Size
        axes[1, 0].scatter(sizes, latencies, s=100, alpha=0.7)
        axes[1, 0].set_xlabel("Model Size (MB)")
        axes[1, 0].set_ylabel("Latency (ms)")
        axes[1, 0].set_title("Latency vs Model Size")
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (sizes[i], latencies[i]))
            
        # Memory Usage
        axes[1, 1].bar(model_names, memories, alpha=0.7)
        axes[1, 1].set_xlabel("Model")
        axes[1, 1].set_ylabel("Memory Usage (MB)")
        axes[1, 1].set_title("Memory Usage Comparison")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
