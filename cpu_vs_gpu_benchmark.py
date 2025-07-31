#!/usr/bin/env python3
"""
CPU vs GPU Benchmark for Mac M4 Chip
This script compares CPU and GPU performance for deep learning tasks on Mac M4.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import platform
import os

print("=" * 70)
print(f"System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print("=" * 70)

# Define model architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

# Generate synthetic data
def generate_data(size=1000, img_size=64, num_classes=10):
    data = torch.randn(size, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (size,))
    return data, labels

# Benchmark training function
def benchmark_training(model, data_loader, device, num_epochs=2):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Warm-up
    model.train()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break
    
    # Benchmark
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader):.4f}')
    
    return time.time() - start_time

# Benchmark inference function
def benchmark_inference(model, data_loader, device):
    model = model.to(device)
    model.eval()
    
    # Warm-up
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)
        break
    
    # Benchmark
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    
    return time.time() - start_time

# Main benchmark function
def run_benchmark():
    # Parameters
    batch_size = 32
    num_epochs = 2
    
    # Generate data
    print("\nGenerating synthetic data...")
    train_data, train_labels = generate_data(size=1000, img_size=64)
    test_data, test_labels = generate_data(size=200, img_size=64)
    
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    results = {}
    
    # CPU Benchmark
    print("\n" + "=" * 30)
    print("Running CPU Benchmark...")
    cpu_device = torch.device("cpu")
    cpu_model = ResNet18()
    
    print("\nTraining on CPU...")
    cpu_train_time = benchmark_training(cpu_model, train_loader, cpu_device, num_epochs)
    print(f"CPU Training Time: {cpu_train_time:.2f} seconds")
    results["cpu_train_time"] = cpu_train_time
    
    print("\nInference on CPU...")
    cpu_inference_time = benchmark_inference(cpu_model, test_loader, cpu_device)
    print(f"CPU Inference Time: {cpu_inference_time:.2f} seconds")
    results["cpu_inference_time"] = cpu_inference_time
    
    # GPU Benchmark (MPS)
    if torch.backends.mps.is_available():
        print("\n" + "=" * 30)
        print("Running GPU (MPS) Benchmark...")
        gpu_device = torch.device("mps")
        gpu_model = ResNet18()
        
        print("\nTraining on GPU (MPS)...")
        gpu_train_time = benchmark_training(gpu_model, train_loader, gpu_device, num_epochs)
        print(f"GPU Training Time: {gpu_train_time:.2f} seconds")
        results["gpu_train_time"] = gpu_train_time
        
        print("\nInference on GPU (MPS)...")
        gpu_inference_time = benchmark_inference(gpu_model, test_loader, gpu_device)
        print(f"GPU Inference Time: {gpu_inference_time:.2f} seconds")
        results["gpu_inference_time"] = gpu_inference_time
        
        # Calculate speedup
        print("\n" + "=" * 30)
        print("Performance Comparison (CPU vs GPU)")
        print(f"Training Speedup: {cpu_train_time / gpu_train_time:.2f}x faster on GPU")
        print(f"Inference Speedup: {cpu_inference_time / gpu_inference_time:.2f}x faster on GPU")
    else:
        print("\nMPS (GPU acceleration) is not available on this system.")
    
    print("\n" + "=" * 30)
    print("Benchmark Complete!")
    return results

if __name__ == "__main__":
    results = run_benchmark() 