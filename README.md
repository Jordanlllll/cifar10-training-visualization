# CIFAR-10 Data Loading and Training Visualization

This project demonstrates how to load CIFAR-10 data using PyTorch data loader, define a neural network using EfficientNetV2 from the torchvision model base, train the network, and visualize the training process and network architecture.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- torchviz

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/githubnext/workspace-blank.git
   cd workspace-blank
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision matplotlib torchviz
   ```

## Usage

1. Load CIFAR-10 data:
   The `data_loader.py` script contains a function to load CIFAR-10 data using PyTorch data loader.

2. Define the neural network:
   The `model.py` script defines a neural network class using EfficientNetV2 from the torchvision model base.

3. Train the network:
   The `train.py` script trains the network and saves a figure to plot the training process, showing both train accuracy and test accuracy vs. epoch.

4. Visualize the network architecture:
   The `visualize.py` script draws a figure to visualize the network architecture.

## Running the Project

1. Load CIFAR-10 data:
   ```bash
   python data_loader.py
   ```

2. Define the neural network:
   ```bash
   python model.py
   ```

3. Train the network:
   ```bash
   python train.py
   ```

4. Visualize the network architecture:
   ```bash
   python visualize.py
   ```

## Details

### CIFAR-10 Data Loading

The `data_loader.py` script uses PyTorch data loader to load CIFAR-10 data. It returns train and test data loaders.

### Neural Network Definition

The `model.py` script defines a neural network class using EfficientNetV2 from the torchvision model base.

### Training Process

The `train.py` script trains the network and saves a figure to plot the training process. It shows both train accuracy and test accuracy vs. epoch.

### Network Architecture Visualization

The `visualize.py` script draws a figure to visualize the network architecture and saves it as a PNG file.
