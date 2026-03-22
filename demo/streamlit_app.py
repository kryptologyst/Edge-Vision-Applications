"""Streamlit demo application for Edge Vision Applications."""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Import our modules
from src.models import EdgeMobileNetV2, create_model
from src.pipelines import ImagePreprocessor, InferencePipeline, CameraCapture
from src.evaluation import PerformanceProfiler, ModelComparator
from src.utils import set_deterministic_seed, get_device, format_model_size


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Edge Vision Applications Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ DISCLAIMER</h4>
    <p><strong>This project is for research and educational purposes only. NOT FOR SAFETY-CRITICAL DEPLOYMENT.</strong></p>
    <p>This demo simulates edge device constraints and performance characteristics. 
    Actual performance may vary significantly on real hardware.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔍 Edge Vision Applications Demo</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["mobilenet_v2", "quantized_mobilenet_v2", "pruned_mobilenet_v2", "distilled_mobilenet_v2"],
    help="Choose the model variant to test"
)

# Device simulation
device_type = st.sidebar.selectbox(
    "Simulate Device",
    ["cpu", "cuda", "raspberry_pi", "jetson_nano", "android_mobile"],
    help="Simulate different edge device constraints"
)

# Input source
input_source = st.sidebar.radio(
    "Input Source",
    ["Upload Image", "Webcam", "Sample Data"],
    help="Choose how to provide input data"
)

# Performance settings
st.sidebar.subheader("Performance Settings")
enable_benchmarking = st.sidebar.checkbox("Enable Benchmarking", value=True)
enable_privacy_mask = st.sidebar.checkbox("Enable Privacy Masking", value=False)
max_frames = st.sidebar.slider("Max Frames (Webcam)", 1, 100, 30)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = {}

# Model initialization
@st.cache_resource
def load_model(model_type: str, device: str):
    """Load and cache model."""
    try:
        set_deterministic_seed(42)
        
        # Create model configuration
        config = {
            "num_classes": 1000,
            "pretrained": True
        }
        
        # Add specific configs based on model type
        if model_type == "quantized_mobilenet_v2":
            config.update({
                "base_config": {"num_classes": 1000, "pretrained": True},
                "quantization_config": {"method": "int8"}
            })
        elif model_type == "pruned_mobilenet_v2":
            config.update({
                "base_config": {"num_classes": 1000, "pretrained": True},
                "pruning_config": {"method": "structured", "sparsity": 0.5}
            })
        elif model_type == "distilled_mobilenet_v2":
            config.update({
                "student_config": {"num_classes": 1000, "pretrained": True},
                "distillation_config": {"temperature": 3.0, "alpha": 0.7}
            })
            
        model = create_model(model_type, config)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load model
if st.session_state.model is None or st.session_state.model_type != model_type:
    with st.spinner(f"Loading {model_type}..."):
        st.session_state.model = load_model(model_type, device_type)
        st.session_state.model_type = model_type
        
if st.session_state.model is None:
    st.error("Failed to load model. Please check the configuration.")
    st.stop()

# Initialize preprocessor and pipeline
if st.session_state.preprocessor is None:
    st.session_state.preprocessor = ImagePreprocessor(
        input_size=(224, 224),
        normalize=True,
        augment=False
    )

if st.session_state.pipeline is None:
    device = get_device()
    st.session_state.pipeline = InferencePipeline(
        model=st.session_state.model,
        preprocessor=st.session_state.preprocessor,
        device=device,
        batch_size=1
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input")
    
    # Handle different input sources
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for classification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Run inference
            if st.button("Classify Image"):
                with st.spinner("Running inference..."):
                    start_time = time.time()
                    results = st.session_state.pipeline.predict_single(image)
                    inference_time = (time.time() - start_time) * 1000
                    
                    st.success(f"Inference completed in {inference_time:.2f} ms")
                    
                    # Display results
                    st.subheader("Classification Results")
                    
                    # Top prediction
                    top_pred = results["top_prediction"]
                    st.metric(
                        "Top Prediction",
                        f"Class {top_pred['class_id']}",
                        f"{top_pred['probability']:.2%}"
                    )
                    
                    # All predictions
                    st.subheader("Top 5 Predictions")
                    for i, pred in enumerate(results["predictions"]):
                        st.progress(pred["probability"])
                        st.write(f"{i+1}. Class {pred['class_id']}: {pred['probability']:.2%}")
                        
    elif input_source == "Webcam":
        st.subheader("Webcam Input")
        
        if st.button("Start Camera"):
            try:
                with CameraCapture(camera_id=0, resolution=(640, 480)) as camera:
                    frame_count = 0
                    placeholder = st.empty()
                    
                    while frame_count < max_frames:
                        frame = camera.read_frame()
                        if frame is None:
                            break
                            
                        # Display frame
                        placeholder.image(frame, caption=f"Frame {frame_count}", use_column_width=True)
                        
                        # Run inference
                        results = st.session_state.pipeline.predict_single(frame)
                        
                        # Display results
                        st.write(f"Frame {frame_count}: Class {results['top_prediction']['class_id']} "
                                f"({results['top_prediction']['probability']:.2%})")
                        
                        frame_count += 1
                        time.sleep(0.1)  # Small delay for visualization
                        
            except Exception as e:
                st.error(f"Camera error: {e}")
                
    else:  # Sample Data
        st.subheader("Sample Data")
        
        if st.button("Generate Sample Data"):
            # Create sample image
            sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            sample_pil = Image.fromarray(sample_image)
            
            st.image(sample_pil, caption="Generated Sample Image", use_column_width=True)
            
            # Run inference
            with st.spinner("Running inference..."):
                results = st.session_state.pipeline.predict_single(sample_pil)
                
                st.success("Inference completed")
                
                # Display results
                st.subheader("Classification Results")
                top_pred = results["top_prediction"]
                st.metric(
                    "Top Prediction",
                    f"Class {top_pred['class_id']}",
                    f"{top_pred['probability']:.2%}"
                )

with col2:
    st.header("Performance Metrics")
    
    # Model information
    st.subheader("Model Information")
    
    if st.session_state.model:
        model_size_info = st.session_state.model.get_model_size()
        
        col_size1, col_size2 = st.columns(2)
        with col_size1:
            st.metric("Parameters", f"{model_size_info['total_parameters']:,}")
        with col_size2:
            st.metric("Model Size", f"{model_size_info['model_size_mb']:.2f} MB")
    
    # Benchmarking
    if enable_benchmarking:
        st.subheader("Performance Benchmark")
        
        if st.button("Run Benchmark"):
            with st.spinner("Running benchmark..."):
                profiler = PerformanceProfiler(st.session_state.model, get_device())
                
                # Latency benchmark
                latency_results = profiler.profile_latency(
                    input_shape=(1, 3, 224, 224),
                    num_runs=50,
                    warmup_runs=5
                )
                
                # Throughput benchmark
                throughput_results = profiler.profile_throughput(
                    input_shape=(1, 3, 224, 224),
                    duration_seconds=5
                )
                
                # Memory benchmark
                memory_results = profiler.profile_memory_usage(
                    input_shape=(1, 3, 224, 224)
                )
                
                # Store results
                st.session_state.benchmark_results = {
                    "latency": latency_results,
                    "throughput": throughput_results,
                    "memory": memory_results
                }
                
                st.success("Benchmark completed!")
    
    # Display benchmark results
    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        # Latency metrics
        st.subheader("Latency Metrics")
        col_lat1, col_lat2, col_lat3 = st.columns(3)
        
        with col_lat1:
            st.metric("Mean Latency", f"{results['latency']['mean_latency_ms']:.2f} ms")
        with col_lat2:
            st.metric("P95 Latency", f"{results['latency']['p95_latency_ms']:.2f} ms")
        with col_lat3:
            st.metric("P99 Latency", f"{results['latency']['p99_latency_ms']:.2f} ms")
        
        # Throughput metrics
        st.subheader("Throughput Metrics")
        col_thr1, col_thr2 = st.columns(2)
        
        with col_thr1:
            st.metric("FPS", f"{results['throughput']['fps']:.2f}")
        with col_thr2:
            st.metric("Inferences/sec", f"{results['throughput']['inferences_per_second']:.2f}")
        
        # Memory metrics
        st.subheader("Memory Metrics")
        col_mem1, col_mem2 = st.columns(2)
        
        with col_mem1:
            st.metric("Memory Used", f"{results['memory']['memory_used_mb']:.2f} MB")
        with col_mem2:
            st.metric("Peak Memory", f"{results['memory']['peak_memory_mb']:.2f} MB")
        
        # Performance visualization
        st.subheader("Performance Visualization")
        
        # Latency distribution
        if 'latency_times' in results['latency']:
            fig_latency = px.histogram(
                x=results['latency']['latency_times'],
                title="Latency Distribution",
                labels={'x': 'Latency (ms)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_latency, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Edge Vision Applications Demo - Research & Educational Use Only</p>
    <p>Built with PyTorch, Streamlit, and optimized for edge deployment</p>
</div>
""", unsafe_allow_html=True)
