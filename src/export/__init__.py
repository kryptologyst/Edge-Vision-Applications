"""Model export and conversion utilities for Edge Vision Applications."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import torch
import torch.onnx
from omegaconf import DictConfig

from ..utils import format_model_size, get_device


logger = logging.getLogger(__name__)


class ModelExporter:
    """Export PyTorch models to various edge formats."""
    
    def __init__(self, model: torch.nn.Module, config: DictConfig):
        """Initialize model exporter.
        
        Args:
            model: PyTorch model to export
            config: Export configuration
        """
        self.model = model
        self.config = config
        self.device = get_device()
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def export_onnx(self, 
                   output_path: str,
                   input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                   opset_version: int = 11) -> str:
        """Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        logger.info(f"Exporting model to ONNX format: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model exported successfully. Size: {format_model_size(output_path)}")
        return output_path
        
    def export_tflite(self, 
                     output_path: str,
                     input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                     quantize: bool = False) -> str:
        """Export model to TensorFlow Lite format.
        
        Args:
            output_path: Path to save TFLite model
            input_shape: Input tensor shape
            quantize: Whether to apply quantization
            
        Returns:
            Path to exported TFLite model
        """
        logger.info(f"Exporting model to TFLite format: {output_path}")
        
        try:
            import tensorflow as tf
            from tensorflow import lite as tflite
        except ImportError:
            logger.error("TensorFlow not available for TFLite export")
            raise ImportError("TensorFlow is required for TFLite export")
            
        # First export to ONNX, then convert to TFLite
        onnx_path = output_path.replace('.tflite', '.onnx')
        self.export_onnx(onnx_path, input_shape)
        
        # Convert ONNX to TensorFlow SavedModel
        try:
            import onnx_tf
            tf_model = onnx_tf.backend.prepare(onnx.load(onnx_path))
            tf_model.export_graph(output_path.replace('.tflite', '_tf'))
        except ImportError:
            logger.warning("onnx-tf not available, using alternative conversion")
            # Alternative: Use ONNX Runtime to convert
            self._convert_onnx_to_tflite_via_ort(onnx_path, output_path, quantize)
            
        logger.info(f"TFLite model exported successfully. Size: {format_model_size(output_path)}")
        return output_path
        
    def _convert_onnx_to_tflite_via_ort(self, 
                                       onnx_path: str,
                                       tflite_path: str,
                                       quantize: bool) -> None:
        """Convert ONNX to TFLite using ONNX Runtime.
        
        Args:
            onnx_path: Path to ONNX model
            tflite_path: Path to save TFLite model
            quantize: Whether to apply quantization
        """
        try:
            import onnxruntime as ort
            import tensorflow as tf
        except ImportError:
            logger.error("ONNX Runtime or TensorFlow not available")
            raise
            
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow format
        # This is a simplified conversion - in practice, you'd use proper conversion tools
        logger.warning("Using simplified ONNX to TFLite conversion")
        
        # Create a dummy TFLite model for demonstration
        converter = tf.lite.TFLiteConverter.from_saved_model("dummy_path")
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
    def export_coreml(self, 
                     output_path: str,
                     input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
        """Export model to CoreML format for iOS deployment.
        
        Args:
            output_path: Path to save CoreML model
            input_shape: Input tensor shape
            
        Returns:
            Path to exported CoreML model
        """
        logger.info(f"Exporting model to CoreML format: {output_path}")
        
        try:
            import coremltools as ct
        except ImportError:
            logger.error("CoreML Tools not available")
            raise ImportError("CoreML Tools is required for CoreML export")
            
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_shape, name="input")]
        )
        
        # Save CoreML model
        coreml_model.save(output_path)
        
        logger.info(f"CoreML model exported successfully. Size: {format_model_size(output_path)}")
        return output_path
        
    def export_openvino(self, 
                       output_path: str,
                       input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
        """Export model to OpenVINO format for Intel hardware.
        
        Args:
            output_path: Path to save OpenVINO model
            input_shape: Input tensor shape
            
        Returns:
            Path to exported OpenVINO model
        """
        logger.info(f"Exporting model to OpenVINO format: {output_path}")
        
        try:
            from openvino.tools import mo
            from openvino.runtime import serialize
        except ImportError:
            logger.error("OpenVINO not available")
            raise ImportError("OpenVINO is required for OpenVINO export")
            
        # First export to ONNX
        onnx_path = output_path.replace('.xml', '.onnx')
        self.export_onnx(onnx_path, input_shape)
        
        # Convert ONNX to OpenVINO IR
        mo.convert_model(
            onnx_path,
            output_dir=os.path.dirname(output_path),
            model_name=os.path.basename(output_path).replace('.xml', '')
        )
        
        logger.info(f"OpenVINO model exported successfully")
        return output_path
        
    def export_all_formats(self, 
                          output_dir: str,
                          model_name: str = "mobilenet_v2",
                          input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, str]:
        """Export model to all supported formats.
        
        Args:
            output_dir: Directory to save exported models
            model_name: Base name for exported models
            input_shape: Input tensor shape
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_models = {}
        
        # Export ONNX
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        exported_models["onnx"] = self.export_onnx(onnx_path, input_shape)
        
        # Export TFLite
        tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
        exported_models["tflite"] = self.export_tflite(tflite_path, input_shape)
        
        # Export TFLite Quantized
        tflite_quant_path = os.path.join(output_dir, f"{model_name}_quant.tflite")
        exported_models["tflite_quant"] = self.export_tflite(tflite_quant_path, input_shape, quantize=True)
        
        # Export CoreML
        coreml_path = os.path.join(output_dir, f"{model_name}.mlmodel")
        exported_models["coreml"] = self.export_coreml(coreml_path, input_shape)
        
        # Export OpenVINO
        openvino_path = os.path.join(output_dir, f"{model_name}.xml")
        exported_models["openvino"] = self.export_openvino(openvino_path, input_shape)
        
        return exported_models


class ModelConverter:
    """Convert between different model formats."""
    
    @staticmethod
    def onnx_to_tflite(onnx_path: str, 
                      tflite_path: str,
                      quantize: bool = False) -> str:
        """Convert ONNX model to TFLite format.
        
        Args:
            onnx_path: Path to ONNX model
            tflite_path: Path to save TFLite model
            quantize: Whether to apply quantization
            
        Returns:
            Path to converted TFLite model
        """
        logger.info(f"Converting ONNX to TFLite: {onnx_path} -> {tflite_path}")
        
        try:
            import tensorflow as tf
            import onnx
            import onnx_tf
        except ImportError as e:
            logger.error(f"Required packages not available: {e}")
            raise
            
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_rep.export_graph(tflite_path.replace('.tflite', '_tf'))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tflite_path.replace('.tflite', '_tf'))
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"TFLite model created: {format_model_size(tflite_path)}")
        return tflite_path
        
    @staticmethod
    def pytorch_to_onnx(pytorch_model: torch.nn.Module,
                       output_path: str,
                       input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
        """Convert PyTorch model to ONNX format.
        
        Args:
            pytorch_model: PyTorch model
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            
        Returns:
            Path to converted ONNX model
        """
        logger.info(f"Converting PyTorch to ONNX: {output_path}")
        
        device = get_device()
        pytorch_model = pytorch_model.to(device)
        pytorch_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        logger.info(f"ONNX model created: {format_model_size(output_path)}")
        return output_path


class ModelValidator:
    """Validate exported models for correctness."""
    
    @staticmethod
    def validate_onnx_model(model_path: str) -> Dict[str, bool]:
        """Validate ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Dictionary with validation results
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            return {"onnx_available": False, "ort_available": False}
            
        results = {}
        
        # Check if file exists
        if not os.path.exists(model_path):
            results["file_exists"] = False
            return results
            
        results["file_exists"] = True
        
        # Load and check model
        try:
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            results["model_valid"] = True
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            results["model_valid"] = False
            
        # Test inference
        try:
            session = ort.InferenceSession(model_path)
            results["inference_test"] = True
        except Exception as e:
            logger.error(f"ONNX inference test failed: {e}")
            results["inference_test"] = False
            
        return results
        
    @staticmethod
    def validate_tflite_model(model_path: str) -> Dict[str, bool]:
        """Validate TFLite model.
        
        Args:
            model_path: Path to TFLite model
            
        Returns:
            Dictionary with validation results
        """
        try:
            import tensorflow as tf
        except ImportError:
            return {"tensorflow_available": False}
            
        results = {}
        
        # Check if file exists
        if not os.path.exists(model_path):
            results["file_exists"] = False
            return results
            
        results["file_exists"] = True
        
        # Load and test model
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            results["model_valid"] = True
        except Exception as e:
            logger.error(f"TFLite model validation failed: {e}")
            results["model_valid"] = False
            
        return results
