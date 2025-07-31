#!/usr/bin/env python3
"""
Transformer CPU vs GPU Benchmark for Mac M4 Chip
This script compares CPU and GPU performance for Transformer models on Mac M4.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import platform
import os
from torch.utils.data import DataLoader, TensorDataset

print("=" * 70)
print(f"System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print("=" * 70)

# Define Transformer architecture components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create a vector of shape (max_seq_length)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be part of the module's state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_length=100, num_classes=10):
        super(TransformerModel, self).__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Final classification layer
        self.fc = nn.Linear(d_model, num_classes)
        
        # Initialize parameters
        self._init_weights()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, src_mask=None):
        # src shape: (batch_size, seq_length)
        
        # Embed the tokens and apply positional encoding
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transpose for transformer input (seq_length, batch_size, d_model)
        src = src.transpose(0, 1)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Use the mean of the output sequence for classification
        output = output.transpose(0, 1).mean(dim=1)
        
        # Pass through final classification layer
        output = self.fc(output)
        
        return output

# Function to generate synthetic sequence data
def generate_sequence_data(batch_size=32, seq_length=50, vocab_size=10000, num_classes=10):
    # Generate random token sequences
    sequences = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Generate random class labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return sequences, labels

# Function to create dataloaders
def create_dataloaders(batch_size=32, train_size=1000, test_size=200, 
                       seq_length=50, vocab_size=10000, num_classes=10):
    
    # Generate training data
    train_data, train_labels = generate_sequence_data(
        batch_size=train_size, seq_length=seq_length, 
        vocab_size=vocab_size, num_classes=num_classes
    )
    
    # Generate test data
    test_data, test_labels = generate_sequence_data(
        batch_size=test_size, seq_length=seq_length, 
        vocab_size=vocab_size, num_classes=num_classes
    )
    
    # Create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Benchmark training function
def benchmark_training(model, data_loader, device, num_epochs=2):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
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
    # Model and data parameters
    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    dim_feedforward = 2048
    seq_length = 50
    num_classes = 10
    batch_size = 32
    num_epochs = 2
    
    # Create dataloaders
    print("\nGenerating synthetic sequence data...")
    train_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        train_size=1000,
        test_size=200,
        seq_length=seq_length,
        vocab_size=vocab_size,
        num_classes=num_classes
    )
    
    results = {}
    
    # CPU Benchmark
    print("\n" + "=" * 50)
    print("Running Transformer CPU Benchmark...")
    cpu_device = torch.device("cpu")
    
    # Create model
    cpu_model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=seq_length,
        num_classes=num_classes
    )
    
    print("\nTransformer model details:")
    print(f"- Encoder layers: {num_encoder_layers}")
    print(f"- Attention heads: {nhead}")
    print(f"- Model dimension: {d_model}")
    print(f"- Feed-forward dimension: {dim_feedforward}")
    print(f"- Sequence length: {seq_length}")
    
    print("\nTraining Transformer on CPU...")
    cpu_train_time = benchmark_training(cpu_model, train_loader, cpu_device, num_epochs)
    print(f"CPU Training Time: {cpu_train_time:.2f} seconds")
    results["cpu_train_time"] = cpu_train_time
    
    print("\nInference with Transformer on CPU...")
    cpu_inference_time = benchmark_inference(cpu_model, test_loader, cpu_device)
    print(f"CPU Inference Time: {cpu_inference_time:.2f} seconds")
    results["cpu_inference_time"] = cpu_inference_time
    
    # GPU Benchmark (MPS)
    if torch.backends.mps.is_available():
        print("\n" + "=" * 50)
        print("Running Transformer GPU (MPS) Benchmark...")
        gpu_device = torch.device("mps")
        
        # Create model
        gpu_model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_length=seq_length,
            num_classes=num_classes
        )
        
        print("\nTraining Transformer on GPU (MPS)...")
        gpu_train_time = benchmark_training(gpu_model, train_loader, gpu_device, num_epochs)
        print(f"GPU Training Time: {gpu_train_time:.2f} seconds")
        results["gpu_train_time"] = gpu_train_time
        
        print("\nInference with Transformer on GPU (MPS)...")
        gpu_inference_time = benchmark_inference(gpu_model, test_loader, gpu_device)
        print(f"GPU Inference Time: {gpu_inference_time:.2f} seconds")
        results["gpu_inference_time"] = gpu_inference_time
        
        # Calculate speedup
        print("\n" + "=" * 50)
        print("Transformer Performance Comparison (CPU vs GPU)")
        print(f"Training Speedup: {cpu_train_time / gpu_train_time:.2f}x faster on GPU")
        print(f"Inference Speedup: {cpu_inference_time / gpu_inference_time:.2f}x faster on GPU")
        
        # Check if attention calculation is the bottleneck
        print("\n" + "=" * 50)
        print("Attention Mechanism Performance Test")
        
        # Test for different sequence lengths
        seq_lengths = [32, 64, 128, 256]
        print("\nTesting attention mechanism with varying sequence lengths:")
        
        for test_seq_len in seq_lengths:
            print(f"\nSequence Length: {test_seq_len}")
            
            # Generate test data with current sequence length
            test_inputs = torch.randint(0, vocab_size, (batch_size, test_seq_len)).to(gpu_device)
            
            # Create a simple single-head attention module for testing
            test_attn = nn.MultiheadAttention(d_model, num_heads=1).to(gpu_device)
            
            # Create test embeddings
            test_embeds = torch.rand(test_seq_len, batch_size, d_model).to(gpu_device)
            
            # Warm-up
            with torch.no_grad():
                _ = test_attn(test_embeds, test_embeds, test_embeds)
            
            # Benchmark attention
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):  # Multiple iterations for more accurate timing
                    _ = test_attn(test_embeds, test_embeds, test_embeds)
            gpu_attn_time = time.time() - start_time
            
            # Move to CPU
            test_attn = test_attn.to(cpu_device)
            test_embeds = test_embeds.to(cpu_device)
            
            # Warm-up on CPU
            with torch.no_grad():
                _ = test_attn(test_embeds, test_embeds, test_embeds)
            
            # Benchmark attention on CPU
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):  # Fewer iterations on CPU since it's slower
                    _ = test_attn(test_embeds, test_embeds, test_embeds)
            cpu_attn_time = time.time() - start_time * 0.1  # Adjust for fewer iterations
            
            print(f"Attention mechanism speedup for seq_len={test_seq_len}: {cpu_attn_time/gpu_attn_time:.2f}x faster on GPU")
    else:
        print("\nMPS (GPU acceleration) is not available on this system.")
    
    print("\n" + "=" * 50)
    print("Transformer Benchmark Complete!")
    
    return results

if __name__ == "__main__":
    results = run_benchmark() 