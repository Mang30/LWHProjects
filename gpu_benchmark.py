#!/usr/bin/env python3
"""
GPU Benchmark for Mac M4 Chip
This script tests deep learning model inference and training speed on Mac's GPU.
"""

import time
import numpy as np
import os
import platform

# Display system information
print("=" * 50)
print(f"System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Python version: {platform.python_version()}")
print("=" * 50)

# Try to import and test TensorFlow
try:
    print("\n--- TensorFlow Test ---")
    import tensorflow as tf
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Available devices: {tf.config.list_physical_devices()}")
    
    # Check if MPS (Metal Performance Shaders) is available for macOS
    print(f"MPS available: {tf.config.list_physical_devices('MPS')}")
    
    # Create and run a simple model with TensorFlow
    print("\nRunning TensorFlow CNN benchmark...")

    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Generate random data
    x_train = np.random.random((1000, 224, 224, 3))
    y_train = np.random.randint(0, 10, (1000,))
    
    # Measure training time
    start_time = time.time()
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=1)
    tf_train_time = time.time() - start_time
    print(f"TensorFlow training time: {tf_train_time:.2f} seconds")
    
    # Measure inference time
    test_data = np.random.random((100, 224, 224, 3))
    start_time = time.time()
    predictions = model.predict(test_data)
    tf_inference_time = time.time() - start_time
    print(f"TensorFlow inference time for 100 images: {tf_inference_time:.2f} seconds")
    
except ImportError:
    print("TensorFlow not installed or not correctly configured.")
except Exception as e:
    print(f"Error running TensorFlow benchmark: {e}")

# Try to import and test PyTorch
try:
    print("\n--- PyTorch Test ---")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Check if MPS is available for macOS
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Set device to MPS if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Create a simple CNN model with PyTorch
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.conv3 = nn.Conv2d(64, 128, 3, 1)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(100352, 128)  # Adjusted for input size
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    print("\nRunning PyTorch CNN benchmark...")
    
    # Instantiate the model and move to device
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Generate random data
    x_train = torch.randn(1000, 3, 224, 224).to(device)
    y_train = torch.randint(0, 10, (1000,)).to(device)
    
    # Measure training time
    start_time = time.time()
    model.train()
    for epoch in range(2):
        print(f"Epoch {epoch+1}")
        running_loss = 0.0
        for i in range(0, len(x_train), 32):
            inputs = x_train[i:i+32]
            labels = y_train[i:i+32]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 320 == 0:
                print(f"Batch {i//32}/{len(x_train)//32}, Loss: {running_loss/(i//32+1):.3f}")
    
    pt_train_time = time.time() - start_time
    print(f"PyTorch training time: {pt_train_time:.2f} seconds")
    
    # Measure inference time
    test_data = torch.randn(100, 3, 224, 224).to(device)
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
    pt_inference_time = time.time() - start_time
    print(f"PyTorch inference time for 100 images: {pt_inference_time:.2f} seconds")
    
except ImportError:
    print("PyTorch not installed or not correctly configured.")
except Exception as e:
    print(f"Error running PyTorch benchmark: {e}")

print("\n--- Benchmark Complete ---")
print("=" * 50) 