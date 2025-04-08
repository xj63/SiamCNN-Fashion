import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """获取可用的计算设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_history(history, save_path=None):
    """绘制训练历史记录"""
    plt.figure(figsize=(12, 4))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history['train_losses'])+1), history['train_losses'], label='Training Loss')
    plt.plot(range(1, len(history['test_losses'])+1), history['test_losses'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['accuracies'])+1), history['accuracies'], label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
