# SiamCNN-Fashion

A siamese CNN network for Fashion MNIST classification using PyTorch.

## Features

- Siamese network architecture for similarity learning
- CNN-based feature extraction
- Trained and evaluated on Fashion MNIST dataset

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
