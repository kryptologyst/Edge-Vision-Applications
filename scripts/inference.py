#!/usr/bin/env python3
"""Inference script for Edge Vision Applications."""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import cv2
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from src.models import create_model
from src.pipelines import ImagePreprocessor, InferencePipeline
from src.utils import set_deterministic_seed, get_device, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Edge Vision Applications - Inference")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input image or directory"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="mobilenet_v2",
        choices=["mobilenet_v2", "quantized_mobilenet_v2", "pruned_mobilenet_v2", "distilled_mobilenet_v2"],
        help="Model type to use"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/device/raspberry_pi.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run performance benchmark"
    )
    
    parser.add_argument(
        "--save-predictions", 
        action="store_true",
        help="Save prediction results to file"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_image(image_path: str) -> Image.Image:
    """Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def process_single_image(pipeline: InferencePipeline, 
                        image_path: str,
                        output_dir: str) -> Dict:
    """Process a single image.
    
    Args:
        pipeline: Inference pipeline
        image_path: Path to image file
        output_dir: Output directory
        
    Returns:
        Dictionary with results
    """
    logger.info(f"Processing image: {image_path}")
    
    # Load image
    image = load_image(image_path)
    
    # Run inference
    start_time = time.time()
    results = pipeline.predict_single(image)
    inference_time = (time.time() - start_time) * 1000
    
    # Add metadata
    results["image_path"] = image_path
    results["inference_time_ms"] = inference_time
    results["image_size"] = image.size
    
    logger.info(f"Inference completed in {inference_time:.2f} ms")
    logger.info(f"Top prediction: Class {results['top_prediction']['class_id']} "
               f"({results['top_prediction']['probability']:.2%})")
    
    return results


def process_directory(pipeline: InferencePipeline,
                     input_dir: str,
                     output_dir: str) -> list:
    """Process all images in a directory.
    
    Args:
        pipeline: Inference pipeline
        input_dir: Input directory path
        output_dir: Output directory
        
    Returns:
        List of results for all images
    """
    logger.info(f"Processing directory: {input_dir}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file_path in Path(input_dir).rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(image_files)} image files")
    
    # Process each image
    all_results = []
    for i, image_path in enumerate(image_files):
        try:
            results = process_single_image(pipeline, image_path, output_dir)
            all_results.append(results)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue
    
    logger.info(f"Completed processing {len(all_results)}/{len(image_files)} images")
    return all_results


def run_benchmark(pipeline: InferencePipeline, 
                  config: OmegaConf) -> Dict:
    """Run performance benchmark.
    
    Args:
        pipeline: Inference pipeline
        config: Configuration object
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Running performance benchmark")
    
    from src.evaluation import PerformanceProfiler
    
    profiler = PerformanceProfiler(pipeline.model, pipeline.device)
    
    # Latency benchmark
    latency_results = profiler.profile_latency(
        input_shape=(1, 3, 224, 224),
        num_runs=100,
        warmup_runs=10
    )
    
    # Throughput benchmark
    throughput_results = profiler.profile_throughput(
        input_shape=(1, 3, 224, 224),
        duration_seconds=10
    )
    
    # Memory benchmark
    memory_results = profiler.profile_memory_usage(
        input_shape=(1, 3, 224, 224)
    )
    
    benchmark_results = {
        "latency": latency_results,
        "throughput": throughput_results,
        "memory": memory_results,
        "config": {
            "model_type": config.get("model", {}).get("type", "unknown"),
            "device": pipeline.device,
            "batch_size": pipeline.batch_size
        }
    }
    
    logger.info("Benchmark completed")
    logger.info(f"Mean latency: {latency_results['mean_latency_ms']:.2f} ms")
    logger.info(f"FPS: {throughput_results['fps']:.2f}")
    logger.info(f"Memory usage: {memory_results['memory_used_mb']:.2f} MB")
    
    return benchmark_results


def save_results(results: list, 
                 benchmark_results: Optional[Dict],
                 output_dir: str,
                 save_predictions: bool = True) -> None:
    """Save inference results.
    
    Args:
        results: List of inference results
        benchmark_results: Benchmark results
        output_dir: Output directory
        save_predictions: Whether to save predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if save_predictions and results:
        # Save predictions to JSON
        import json
        
        predictions_file = os.path.join(output_dir, "predictions.json")
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Predictions saved to {predictions_file}")
        
        # Save summary
        summary_file = os.path.join(output_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("Edge Vision Applications - Inference Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total images processed: {len(results)}\n")
            
            if results:
                avg_inference_time = np.mean([r["inference_time_ms"] for r in results])
                f.write(f"Average inference time: {avg_inference_time:.2f} ms\n")
                
                # Count predictions by class
                class_counts = {}
                for result in results:
                    class_id = result["top_prediction"]["class_id"]
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                
                f.write(f"Unique classes detected: {len(class_counts)}\n")
                f.write("Top 5 classes:\n")
                for class_id, count in sorted(class_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]:
                    f.write(f"  Class {class_id}: {count} images\n")
        
        logger.info(f"Summary saved to {summary_file}")
    
    if benchmark_results:
        # Save benchmark results
        benchmark_file = os.path.join(output_dir, "benchmark.json")
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {benchmark_file}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Set random seed
    set_deterministic_seed(args.seed)
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Model type: {args.model}")
    
    # Create model
    model_config = {
        "num_classes": 1000,
        "pretrained": True
    }
    
    # Add specific configs based on model type
    if args.model == "quantized_mobilenet_v2":
        model_config.update({
            "base_config": {"num_classes": 1000, "pretrained": True},
            "quantization_config": {"method": "int8"}
        })
    elif args.model == "pruned_mobilenet_v2":
        model_config.update({
            "base_config": {"num_classes": 1000, "pretrained": True},
            "pruning_config": {"method": "structured", "sparsity": 0.5}
        })
    elif args.model == "distilled_mobilenet_v2":
        model_config.update({
            "student_config": {"num_classes": 1000, "pretrained": True},
            "distillation_config": {"temperature": 3.0, "alpha": 0.7}
        })
    
    model = create_model(args.model, model_config)
    
    # Create preprocessor and pipeline
    preprocessor = ImagePreprocessor(
        input_size=(224, 224),
        normalize=True,
        augment=False
    )
    
    pipeline = InferencePipeline(
        model=model,
        preprocessor=preprocessor,
        device=device,
        batch_size=args.batch_size
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        results = [process_single_image(pipeline, args.input, args.output)]
    elif os.path.isdir(args.input):
        # Directory
        results = process_directory(pipeline, args.input, args.output)
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return
    
    # Run benchmark if requested
    benchmark_results = None
    if args.benchmark:
        benchmark_results = run_benchmark(pipeline, config)
    
    # Save results
    save_results(results, benchmark_results, args.output, args.save_predictions)
    
    logger.info("Inference completed successfully")


if __name__ == "__main__":
    main()
