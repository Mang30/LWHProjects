#!/usr/bin/env python3
"""
Single-Cell RNA-seq GCN CPU vs GPU Benchmark
This script compares CPU and GPU performance for Graph Convolutional Networks on single-cell data.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import scanpy as sc
import platform
import os
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import anndata
from tqdm import tqdm

print("=" * 70)
print(f"System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
try:
    import torch_geometric
    print(f"PyTorch Geometric: {torch_geometric.__version__}")
except ImportError:
    print("PyTorch Geometric not installed")
print("=" * 70)

# Define GCN model for single-cell data
class SCGCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(SCGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x, edge_index):
        # First Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second Graph Convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third Graph Convolution
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Final classification layer
        x = self.lin(x)
        
        return x

# Generate synthetic single-cell RNA-seq data
def generate_synthetic_scdata(num_cells=30000, num_genes=2000, num_classes=10):
    print(f"\nGenerating synthetic single-cell data with {num_cells} cells and {num_genes} genes...")
    
    # Generate gene expression matrix with realistic properties
    # (sparse, zero-inflated, with cluster structure)
    expression = np.random.negative_binomial(
        n=1,
        p=0.1,
        size=(num_cells, num_genes)
    )
    
    # Make it more sparse (zero-inflated)
    mask = np.random.rand(num_cells, num_genes) < 0.8
    expression[mask] = 0
    
    # Generate cell labels (clusters)
    labels = np.random.randint(0, num_classes, num_cells)
    
    # Make cells in same cluster have more similar expression
    for i in range(num_classes):
        cells_in_cluster = np.where(labels == i)[0]
        if len(cells_in_cluster) > 0:
            cluster_factor = np.random.rand(num_genes) * 5  # Random factor per gene
            expression[cells_in_cluster] = expression[cells_in_cluster] * cluster_factor
    
    # Create AnnData object
    adata = anndata.AnnData(X=expression)
    adata.obs['labels'] = labels
    
    print(f"Generated data matrix shape: {adata.X.shape}")
    return adata

# Process single-cell data and create graph
def process_data(adata, n_neighbors=10):
    print(f"\nProcessing data and creating cell graph with {n_neighbors} nearest neighbors...")
    
    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # PCA for dimensionality reduction
    sc.pp.pca(adata, n_comps=50)
    
    # Construct KNN graph
    knn_graph = kneighbors_graph(
        adata.obsm['X_pca'], 
        n_neighbors=n_neighbors, 
        mode='connectivity', 
        include_self=False
    )
    
    # Convert to edge_index format
    adj_coo = knn_graph.tocoo()
    edge_index = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long)
    
    # Extract features and labels
    features = torch.tensor(adata.obsm['X_pca'], dtype=torch.float)
    labels = torch.tensor(adata.obs['labels'].values, dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    print(f"Graph created with {data.num_nodes} nodes and {data.num_edges} edges")
    return data

# Benchmark training function
def benchmark_training(model, data, device, num_epochs=50):
    model = model.to(device)
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Warm-up
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output, data.y)
    loss.backward()
    optimizer.step()
    
    # Benchmark
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    return time.time() - start_time

# Benchmark inference function
def benchmark_inference(model, data, device, num_runs=100):
    model = model.to(device)
    data = data.to(device)
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        _ = model(data.x, data.edge_index)
    
    # Benchmark
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(data.x, data.edge_index)
    
    return time.time() - start_time

# Main benchmark function
def run_benchmark():
    # Parameters
    num_cells = 3000
    num_genes = 2000
    hidden_channels = 128
    num_classes = 10
    num_epochs = 50
    inference_runs = 100
    
    # Generate and process data
    adata = generate_synthetic_scdata(num_cells, num_genes, num_classes)
    data = process_data(adata)
    
    results = {}
    
    # CPU Benchmark
    print("\n" + "=" * 30)
    print("Running GCN CPU Benchmark...")
    cpu_device = torch.device("cpu")
    cpu_model = SCGCN(data.num_features, hidden_channels, num_classes)
    
    print("\nTraining GCN on CPU...")
    cpu_train_time = benchmark_training(cpu_model, data, cpu_device, num_epochs)
    print(f"CPU Training Time: {cpu_train_time:.2f} seconds")
    results["cpu_train_time"] = cpu_train_time
    
    print("\nInference on CPU...")
    cpu_inference_time = benchmark_inference(cpu_model, data, cpu_device, inference_runs)
    print(f"CPU Inference Time ({inference_runs} runs): {cpu_inference_time:.2f} seconds")
    print(f"Average inference time per run: {cpu_inference_time/inference_runs*1000:.2f} ms")
    results["cpu_inference_time"] = cpu_inference_time
    
    # GPU Benchmark (CUDA or MPS)
    if torch.cuda.is_available():
        print("\n" + "=" * 30)
        print("Running GCN GPU (CUDA) Benchmark...")
        gpu_device = torch.device("cuda")
        gpu_model = SCGCN(data.num_features, hidden_channels, num_classes)
        
        print("\nTraining GCN on GPU (CUDA)...")
        gpu_train_time = benchmark_training(gpu_model, data, gpu_device, num_epochs)
        print(f"GPU Training Time: {gpu_train_time:.2f} seconds")
        results["gpu_train_time"] = gpu_train_time
        
        print("\nInference on GPU (CUDA)...")
        gpu_inference_time = benchmark_inference(gpu_model, data, gpu_device, inference_runs)
        print(f"GPU Inference Time ({inference_runs} runs): {gpu_inference_time:.2f} seconds")
        print(f"Average inference time per run: {gpu_inference_time/inference_runs*1000:.2f} ms")
        results["gpu_inference_time"] = gpu_inference_time
        
        # Calculate speedup
        print("\n" + "=" * 30)
        print("Performance Comparison (CPU vs GPU)")
        print(f"Training Speedup: {cpu_train_time / gpu_train_time:.2f}x faster on GPU")
        print(f"Inference Speedup: {cpu_inference_time / gpu_inference_time:.2f}x faster on GPU")
    
    elif torch.backends.mps.is_available():
        print("\n" + "=" * 30)
        print("Running GCN GPU (MPS) Benchmark...")
        gpu_device = torch.device("mps")
        gpu_model = SCGCN(data.num_features, hidden_channels, num_classes)
        
        print("\nTraining GCN on GPU (MPS)...")
        gpu_train_time = benchmark_training(gpu_model, data, gpu_device, num_epochs)
        print(f"GPU Training Time: {gpu_train_time:.2f} seconds")
        results["gpu_train_time"] = gpu_train_time
        
        print("\nInference on GPU (MPS)...")
        gpu_inference_time = benchmark_inference(gpu_model, data, gpu_device, inference_runs)
        print(f"GPU Inference Time ({inference_runs} runs): {gpu_inference_time:.2f} seconds")
        print(f"Average inference time per run: {gpu_inference_time/inference_runs*1000:.2f} ms")
        results["gpu_inference_time"] = gpu_inference_time
        
        # Calculate speedup
        print("\n" + "=" * 30)
        print("Performance Comparison (CPU vs GPU)")
        print(f"Training Speedup: {cpu_train_time / gpu_train_time:.2f}x faster on GPU")
        print(f"Inference Speedup: {cpu_inference_time / gpu_inference_time:.2f}x faster on GPU")
    else:
        print("\nNo GPU acceleration (CUDA or MPS) is available on this system.")
    
    print("\n" + "=" * 30)
    print("Single-Cell GCN Benchmark Complete!")
    
    return results

if __name__ == "__main__":
    results = run_benchmark()