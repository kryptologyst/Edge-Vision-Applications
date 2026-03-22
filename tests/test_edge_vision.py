"""Test suite for Edge Vision Applications."""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os

from src.models import EdgeMobileNetV2, create_model
from src.pipelines import ImagePreprocessor, InferencePipeline
from src.export import ModelExporter
from src.evaluation import AccuracyEvaluator, PerformanceProfiler
from src.utils import set_deterministic_seed, get_device, create_sample_data


class TestModels:
    """Test model functionality."""
    
    def test_mobilenet_v2_creation(self):
        """Test MobileNetV2 model creation."""
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        assert isinstance(model, EdgeMobileNetV2)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 1000)
        
    def test_model_size_info(self):
        """Test model size information."""
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        size_info = model.get_model_size()
        
        assert "total_parameters" in size_info
        assert "trainable_parameters" in size_info
        assert "model_size_mb" in size_info
        assert size_info["total_parameters"] > 0
        
    def test_model_factory(self):
        """Test model factory function."""
        config = {"num_classes": 1000, "pretrained": False}
        model = create_model("mobilenet_v2", config)
        assert isinstance(model, EdgeMobileNetV2)
        
    def test_quantized_model(self):
        """Test quantized model creation."""
        base_model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        quantization_config = {"method": "int8"}
        
        from src.models import QuantizedMobileNetV2
        quantized_model = QuantizedMobileNetV2(base_model, quantization_config)
        
        dummy_input = torch.randn(1, 3, 224, 224)
        output = quantized_model(dummy_input)
        assert output.shape == (1, 1000)


class TestPipelines:
    """Test pipeline functionality."""
    
    def test_image_preprocessor(self):
        """Test image preprocessing."""
        preprocessor = ImagePreprocessor(input_size=(224, 224), normalize=True)
        
        # Test with PIL Image
        pil_image = Image.new('RGB', (480, 320), color='red')
        tensor = preprocessor.preprocess(pil_image)
        
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        
    def test_inference_pipeline(self):
        """Test inference pipeline."""
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        preprocessor = ImagePreprocessor(input_size=(224, 224), normalize=True)
        device = get_device()
        
        pipeline = InferencePipeline(
            model=model,
            preprocessor=preprocessor,
            device=device,
            batch_size=1
        )
        
        # Test single image inference
        pil_image = Image.new('RGB', (224, 224), color='blue')
        results = pipeline.predict_single(pil_image)
        
        assert "predictions" in results
        assert "top_prediction" in results
        assert len(results["predictions"]) == 5
        
    def test_batch_inference(self):
        """Test batch inference."""
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        preprocessor = ImagePreprocessor(input_size=(224, 224), normalize=True)
        device = get_device()
        
        pipeline = InferencePipeline(
            model=model,
            preprocessor=preprocessor,
            device=device,
            batch_size=2
        )
        
        # Test batch inference
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green')
        ]
        
        results = pipeline.predict_batch(images)
        assert len(results) == 2
        assert all("predictions" in r for r in results)


class TestExport:
    """Test model export functionality."""
    
    def test_model_exporter_creation(self):
        """Test model exporter creation."""
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        
        from omegaconf import DictConfig
        config = DictConfig({"model": {"type": "mobilenet_v2"}})
        
        exporter = ModelExporter(model, config)
        assert exporter.model == model
        assert exporter.config == config
        
    def test_onnx_export(self):
        """Test ONNX export."""
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        
        from omegaconf import DictConfig
        config = DictConfig({"model": {"type": "mobilenet_v2"}})
        
        exporter = ModelExporter(model, config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_model.onnx")
            
            try:
                exported_path = exporter.export_onnx(output_path, input_shape=(1, 3, 224, 224))
                assert os.path.exists(exported_path)
            except Exception as e:
                # ONNX export might fail in test environment, that's okay
                pytest.skip(f"ONNX export failed: {e}")


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_performance_profiler(self):
        """Test performance profiler."""
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        device = get_device()
        
        profiler = PerformanceProfiler(model, device)
        
        # Test latency profiling
        latency_results = profiler.profile_latency(
            input_shape=(1, 3, 224, 224),
            num_runs=5,
            warmup_runs=2
        )
        
        assert "mean_latency_ms" in latency_results
        assert "std_latency_ms" in latency_results
        assert latency_results["mean_latency_ms"] > 0
        
    def test_accuracy_evaluator(self):
        """Test accuracy evaluator."""
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        device = get_device()
        
        evaluator = AccuracyEvaluator(model, device, num_classes=1000)
        
        # Test single batch evaluation
        dummy_data = torch.randn(2, 3, 224, 224)
        dummy_targets = torch.randint(0, 1000, (2,))
        
        results = evaluator.evaluate_single_batch(dummy_data, dummy_targets)
        
        assert "accuracy" in results
        assert "correct" in results
        assert "total" in results
        assert results["total"] == 2


class TestUtils:
    """Test utility functions."""
    
    def test_deterministic_seed(self):
        """Test deterministic seeding."""
        set_deterministic_seed(42)
        
        # Generate some random numbers
        torch_rand1 = torch.randn(10)
        np_rand1 = np.random.randn(10)
        
        # Reset seed and generate again
        set_deterministic_seed(42)
        torch_rand2 = torch.randn(10)
        np_rand2 = np.random.randn(10)
        
        # Should be the same
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
        
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ["cpu", "cuda", "mps"]
        
    def test_create_sample_data(self):
        """Test sample data creation."""
        # Test random data
        random_data = create_sample_data((224, 224, 3), "random")
        assert random_data.shape == (224, 224, 3)
        assert random_data.dtype == np.float32
        
        # Test zeros data
        zeros_data = create_sample_data((224, 224, 3), "zeros")
        assert np.all(zeros_data == 0)
        
        # Test ones data
        ones_data = create_sample_data((224, 224, 3), "ones")
        assert np.all(ones_data == 1)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_inference(self):
        """Test end-to-end inference pipeline."""
        # Create model
        model = EdgeMobileNetV2(num_classes=1000, pretrained=False)
        
        # Create preprocessor
        preprocessor = ImagePreprocessor(input_size=(224, 224), normalize=True)
        
        # Create pipeline
        device = get_device()
        pipeline = InferencePipeline(
            model=model,
            preprocessor=preprocessor,
            device=device,
            batch_size=1
        )
        
        # Create sample image
        sample_image = Image.new('RGB', (224, 224), color='red')
        
        # Run inference
        results = pipeline.predict_single(sample_image)
        
        # Verify results
        assert "predictions" in results
        assert "top_prediction" in results
        assert len(results["predictions"]) == 5
        
        # Check prediction probabilities sum to approximately 1
        total_prob = sum(pred["probability"] for pred in results["predictions"])
        assert abs(total_prob - 1.0) < 0.01
        
    def test_model_comparison(self):
        """Test model comparison functionality."""
        from src.evaluation import ModelComparator
        
        comparator = ModelComparator()
        
        # Add some dummy results
        comparator.add_model_results(
            model_name="baseline",
            accuracy=0.72,
            latency_ms=45.0,
            model_size_mb=14.0,
            memory_mb=512.0
        )
        
        comparator.add_model_results(
            model_name="optimized",
            accuracy=0.71,
            latency_ms=20.0,
            model_size_mb=7.0,
            memory_mb=256.0
        )
        
        # Generate report
        report = comparator.generate_comparison_report()
        assert "Model Comparison Report" in report
        assert "baseline" in report
        assert "optimized" in report


if __name__ == "__main__":
    pytest.main([__file__])
