# Mac M4 GPU Benchmark

This project contains benchmark scripts for testing deep learning model performance on Mac M4 chips using the GPU acceleration via Metal Performance Shaders (MPS).

## Setup

1. First, install the required dependencies:

```bash
pip install -r requirements.txt
```

2. For TensorFlow support on Apple Silicon (M-series), you may need to install the tensorflow-metal plugin:

```bash
pip install tensorflow-metal
```

## Running the Benchmarks

### Full Framework Benchmark

This script tests both TensorFlow and PyTorch frameworks:

```bash
python gpu_benchmark.py
```

### CPU vs GPU Direct Comparison

This script provides a direct comparison between CPU and GPU performance using PyTorch with ResNet18:

```bash
python cpu_vs_gpu_benchmark.py
```

### Transformer Model Benchmark

This script specifically tests Transformer model performance, which is particularly relevant for NLP tasks:

```bash
python transformer_benchmark.py
```

## What the Benchmarks Test

### Full Framework Benchmark (`gpu_benchmark.py`)

1. **System Information**: Displays your system's hardware and software configuration.

2. **TensorFlow Tests**:
   - Verifies TensorFlow installation and GPU availability
   - Trains a CNN model on synthetic data and measures training time
   - Runs inference on synthetic data and measures inference time

3. **PyTorch Tests**:
   - Verifies PyTorch installation and MPS (Metal Performance Shaders) availability
   - Trains a CNN model on synthetic data using MPS acceleration
   - Runs inference on synthetic data and measures inference time

### CPU vs GPU Comparison (`cpu_vs_gpu_benchmark.py`)

This script focuses on providing a direct comparison between CPU and GPU performance:

1. **ResNet18 Architecture**: Uses a more complex and realistic model architecture
2. **Training Performance**: Measures the time taken to train the model on both CPU and GPU
3. **Inference Performance**: Measures the time taken for inference on both CPU and GPU
4. **Speedup Ratio**: Calculates how many times faster the GPU is compared to the CPU

### Transformer Benchmark (`transformer_benchmark.py`)

This script tests Transformer architecture performance, which is especially relevant for NLP tasks and modern foundation models:

1. **Transformer Architecture**: Implements a standard Transformer encoder with self-attention
2. **Sequence Processing**: Tests performance on variable-length sequence data
3. **Attention Mechanism Analysis**: Specifically measures self-attention performance at different sequence lengths
4. **Scaling Properties**: Shows how GPU acceleration scales with increasing sequence length
5. **Training and Inference**: Measures both training and inference times on CPU vs GPU

## Results Interpretation

The benchmarks output timing information for both training and inference phases. The results show how well your M4 chip's GPU accelerates deep learning workloads compared to CPU-only processing.

If MPS is detected and utilized properly, you should see significantly faster processing compared to CPU-only execution. The `cpu_vs_gpu_benchmark.py` and `transformer_benchmark.py` scripts will explicitly show you the speedup ratio.

Typical speedups on Apple Silicon for deep learning tasks range from 3x to 10x depending on the specific model architecture and operations. Transformer models with self-attention often show even higher speedups on GPU since attention operations are highly parallelizable.

## Troubleshooting

If you encounter issues with GPU acceleration:

- Make sure you have the latest version of macOS
- Ensure you're using the latest versions of TensorFlow and PyTorch with proper Metal support
- For TensorFlow, ensure tensorflow-metal is properly installed
- For PyTorch, ensure your version supports MPS (versions 1.12+ should have this support) 