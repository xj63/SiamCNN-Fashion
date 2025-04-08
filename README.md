# SiamCNN-Fashion

A siamese CNN network for Fashion MNIST classification using PyTorch.

See Website <https://simacnn-fashion.xj63.fun>

## Features

- Siamese network architecture for similarity learning
- CNN-based feature extraction
- Trained and evaluated on Fashion MNIST dataset

## Available Model Versions

This repository contains multiple versions of our Siamese CNN model for Fashion MNIST similarity learning. You can access different implementations through the [tags](https://github.com/xj63/SiamCNN-Fashion/tags).

### [large-model](https://github.com/xj63/SiamCNN-Fashion/tree/large-model)
Our most comprehensive implementation featuring:
- Three convolutional layers for feature extraction
- Higher-dimensional feature embeddings (128-D)
- Fully connected layers for similarity judgment

### [small-model](https://github.com/xj63/SiamCNN-Fashion/tree/small-model)
A more efficient version based on the large model with:
- Reduced architecture with two convolutional layers
- Lower-dimensional feature embeddings (64-D)

### [euclidean-distance](https://github.com/xj63/SiamCNN-Fashion/tree/euclidean-distance)
Our current baseline implementation that:
- Builds on the small-model architecture
- Replaces the fully connected layer similarity judgment with direct Euclidean distance measurement
- Provides a more interpretable similarity metric
- Requires fewer parameters and is more generalizable to unseen classes
- **This is the version featured in the main documentation**

Each version includes complete training code.

## Installation

```bash
# Clone the repository
git clone https://github.com/xj63/SiamCnn-Fashion.git
cd SiamCnn-Fashion
```

## Usage

```bash
# Run with default parameters
uv run main.py

# Run with custom parameters
uv run main.py --epochs 20 --batch_size 128 --learning_rate 0.0005
```

## Training Results

After training, you can find:
- Saved model at `./outputs/siamese_model.pth`
- Training history plot at `./outputs/training_history.png`

## Parameters

- `--data_dir`: Directory to store datasets (default: ./data)
- `--output_dir`: Directory to save outputs (default: ./outputs)
- `--train_pairs`: Number of training pairs (default: 10000)
- `--test_pairs`: Number of test pairs (default: 2000)
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 10)
- `--learning_rate`: Learning rate (default: 0.001)
- `--seed`: Random seed (default: 42)

## Generate Website Assets

```bash
uv run main.py
mv ./outputs/**/*.png ./website/assets/
```
