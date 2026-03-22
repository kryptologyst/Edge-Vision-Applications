#!/usr/bin/env python3
"""Model export script for Edge Vision Applications."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf

from src.models import create_model
from src.export import ModelExporter, ModelValidator
from src.utils import set_deterministic_seed, get_device, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Edge Vision Applications - Model Export")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="mobilenet_v2",
        choices=["mobilenet_v2", "quantized_mobilenet_v2", "pruned_mobilenet_v2", "distilled_mobilenet_v2"],
        help="Model type to export"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/device/raspberry_pi.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="exported_models",
        help="Output directory for exported models"
    )
    
    parser.add_argument(
        "--targets", 
        nargs="+",
        default=["onnx", "tflite", "coreml"],
        choices=["onnx", "tflite", "tflite_quant", "coreml", "openvino"],
        help="Target formats to export"
    )
    
    parser.add_argument(
        "--input-shape", 
        nargs=4,
        type=int,
        default=[1, 3, 224, 224],
        metavar=("BATCH", "CHANNELS", "HEIGHT", "WIDTH"),
        help="Input shape for export"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate exported models"
    )
    
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Benchmark exported models"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def create_model_config(model_type: str) -> Dict:
    """Create model configuration based on type.
    
    Args:
        model_type: Type of model to create
        
    Returns:
        Model configuration dictionary
    """
    base_config = {
        "num_classes": 1000,
        "pretrained": True
    }
    
    if model_type == "quantized_mobilenet_v2":
        base_config.update({
            "base_config": {"num_classes": 1000, "pretrained": True},
            "quantization_config": {"method": "int8"}
        })
    elif model_type == "pruned_mobilenet_v2":
        base_config.update({
            "base_config": {"num_classes": 1000, "pretrained": True},
            "pruning_config": {"method": "structured", "sparsity": 0.5}
        })
    elif model_type == "distilled_mobilenet_v2":
        base_config.update({
            "student_config": {"num_classes": 1000, "pretrained": True},
            "distillation_config": {"temperature": 3.0, "alpha": 0.7}
        })
    
    return base_config


def export_model(model_type: str,
                 output_dir: str,
                 targets: List[str],
                 input_shape: tuple,
                 config: OmegaConf) -> Dict[str, str]:
    """Export model to specified formats.
    
    Args:
        model_type: Type of model to export
        output_dir: Output directory
        targets: List of target formats
        input_shape: Input tensor shape
        config: Configuration object
        
    Returns:
        Dictionary mapping format names to file paths
    """
    logger = logging.getLogger(__name__)
    
    # Create model
    model_config = create_model_config(model_type)
    model = create_model(model_type, model_config)
    
    # Create exporter
    exporter = ModelExporter(model, config)
    
    # Export to all formats
    exported_models = exporter.export_all_formats(
        output_dir=output_dir,
        model_name=model_type,
        input_shape=input_shape
    )
    
    # Filter to requested targets
    filtered_models = {}
    for target in targets:
        if target in exported_models:
            filtered_models[target] = exported_models[target]
        else:
            logger.warning(f"Target format {target} not available")
    
    return filtered_models


def validate_models(exported_models: Dict[str, str]) -> Dict[str, Dict[str, bool]]:
    """Validate exported models.
    
    Args:
        exported_models: Dictionary of exported model paths
        
    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger(__name__)
    validation_results = {}
    
    for format_name, model_path in exported_models.items():
        logger.info(f"Validating {format_name} model: {model_path}")
        
        if format_name in ["onnx"]:
            validation_results[format_name] = ModelValidator.validate_onnx_model(model_path)
        elif format_name in ["tflite", "tflite_quant"]:
            validation_results[format_name] = ModelValidator.validate_tflite_model(model_path)
        else:
            logger.warning(f"Validation not implemented for {format_name}")
            validation_results[format_name] = {"validation_available": False}
    
    return validation_results


def benchmark_models(exported_models: Dict[str, str],
                    input_shape: tuple) -> Dict[str, Dict]:
    """Benchmark exported models.
    
    Args:
        exported_models: Dictionary of exported model paths
        input_shape: Input tensor shape
        
    Returns:
        Dictionary with benchmark results
    """
    logger = logging.getLogger(__name__)
    benchmark_results = {}
    
    for format_name, model_path in exported_models.items():
        logger.info(f"Benchmarking {format_name} model")
        
        try:
            if format_name == "onnx":
                benchmark_results[format_name] = benchmark_onnx_model(model_path, input_shape)
            elif format_name in ["tflite", "tflite_quant"]:
                benchmark_results[format_name] = benchmark_tflite_model(model_path, input_shape)
            else:
                logger.warning(f"Benchmarking not implemented for {format_name}")
                
        except Exception as e:
            logger.error(f"Failed to benchmark {format_name}: {e}")
            benchmark_results[format_name] = {"error": str(e)}
    
    return benchmark_results


def benchmark_onnx_model(model_path: str, input_shape: tuple) -> Dict:
    """Benchmark ONNX model.
    
    Args:
        model_path: Path to ONNX model
        input_shape: Input tensor shape
        
    Returns:
        Benchmark results
    """
    import time
    import numpy as np
    
    try:
        import onnxruntime as ort
        
        # Create inference session
        session = ort.InferenceSession(model_path)
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {"input": dummy_input})
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            session.run(None, {"input": dummy_input})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            "mean_latency_ms": float(np.mean(times)),
            "std_latency_ms": float(np.std(times)),
            "p95_latency_ms": float(np.percentile(times, 95)),
            "min_latency_ms": float(np.min(times)),
            "max_latency_ms": float(np.max(times)),
        }
        
    except ImportError:
        return {"error": "ONNX Runtime not available"}


def benchmark_tflite_model(model_path: str, input_shape: tuple) -> Dict:
    """Benchmark TFLite model.
    
    Args:
        model_path: Path to TFLite model
        input_shape: Input tensor shape
        
    Returns:
        Benchmark results
    """
    import time
    import numpy as np
    
    try:
        import tensorflow as tf
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            "mean_latency_ms": float(np.mean(times)),
            "std_latency_ms": float(np.std(times)),
            "p95_latency_ms": float(np.percentile(times, 95)),
            "min_latency_ms": float(np.min(times)),
            "max_latency_ms": float(np.max(times)),
        }
        
    except ImportError:
        return {"error": "TensorFlow not available"}


def save_export_results(exported_models: Dict[str, str],
                       validation_results: Optional[Dict[str, Dict[str, bool]]],
                       benchmark_results: Optional[Dict[str, Dict]],
                       output_dir: str) -> None:
    """Save export results to files.
    
    Args:
        exported_models: Dictionary of exported model paths
        validation_results: Validation results
        benchmark_results: Benchmark results
        output_dir: Output directory
    """
    import json
    
    # Save export summary
    summary_file = os.path.join(output_dir, "export_summary.json")
    
    summary = {
        "exported_models": exported_models,
        "validation_results": validation_results,
        "benchmark_results": benchmark_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Export summary saved to {summary_file}")
    
    # Save human-readable report
    report_file = os.path.join(output_dir, "export_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("Edge Vision Applications - Model Export Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Exported Models:\n")
        f.write("-" * 20 + "\n")
        for format_name, model_path in exported_models.items():
            f.write(f"{format_name}: {model_path}\n")
        
        if validation_results:
            f.write("\nValidation Results:\n")
            f.write("-" * 20 + "\n")
            for format_name, results in validation_results.items():
                f.write(f"{format_name}:\n")
                for key, value in results.items():
                    f.write(f"  {key}: {value}\n")
        
        if benchmark_results:
            f.write("\nBenchmark Results:\n")
            f.write("-" * 20 + "\n")
            for format_name, results in benchmark_results.items():
                f.write(f"{format_name}:\n")
                if "error" in results:
                    f.write(f"  Error: {results['error']}\n")
                else:
                    f.write(f"  Mean Latency: {results['mean_latency_ms']:.2f} ms\n")
                    f.write(f"  P95 Latency: {results['p95_latency_ms']:.2f} ms\n")
    
    logger.info(f"Export report saved to {report_file}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Set random seed
    set_deterministic_seed(args.seed)
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Exporting {args.model} to {args.targets}")
    logger.info(f"Input shape: {args.input_shape}")
    
    # Export model
    exported_models = export_model(
        model_type=args.model,
        output_dir=args.output_dir,
        targets=args.targets,
        input_shape=tuple(args.input_shape),
        config=config
    )
    
    logger.info(f"Exported {len(exported_models)} model formats")
    
    # Validate models if requested
    validation_results = None
    if args.validate:
        validation_results = validate_models(exported_models)
        logger.info("Model validation completed")
    
    # Benchmark models if requested
    benchmark_results = None
    if args.benchmark:
        benchmark_results = benchmark_models(exported_models, tuple(args.input_shape))
        logger.info("Model benchmarking completed")
    
    # Save results
    save_export_results(exported_models, validation_results, benchmark_results, args.output_dir)
    
    logger.info("Model export completed successfully")


if __name__ == "__main__":
    main()
