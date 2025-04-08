import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_data_transforms():
    """定义数据预处理转换"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def load_fashion_mnist(root='./data'):
    """加载Fashion MNIST数据集"""
    transform = get_data_transforms()

    train_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset

class SiameseDataset(Dataset):
    """创建孪生网络数据集"""
    def __init__(self, dataset, num_pairs=10000):
        self.dataset = dataset
        self.num_pairs = num_pairs
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        pairs = []
        labels = []

        # 获取各个类别的索引
        indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in indices:
                indices[label] = []
            indices[label].append(idx)

        # 生成正样本对（相同类别的两个样本）
        for _ in range(self.num_pairs // 2):
            label = np.random.choice(list(indices.keys()))
            if len(indices[label]) >= 2:
                idx1, idx2 = np.random.choice(indices[label], 2, replace=False)
                pairs.append((idx1, idx2))
                labels.append(1)  # 1表示相似

        # 生成负样本对（不同类别的两个样本）
        for _ in range(self.num_pairs // 2):
            label1, label2 = np.random.choice(list(indices.keys()), 2, replace=False)
            idx1 = np.random.choice(indices[label1])
            idx2 = np.random.choice(indices[label2])
            pairs.append((idx1, idx2))
            labels.append(0)  # 0表示不相似

        return list(zip(pairs, labels))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (idx1, idx2), label = self.pairs[idx]
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]
        return img1, img2, torch.tensor(label, dtype=torch.float32)

def create_data_loaders(train_dataset, test_dataset, batch_size=64, train_pairs=10000, test_pairs=2000):
    """创建训练和测试数据加载器"""
    train_siamese_dataset = SiameseDataset(train_dataset, num_pairs=train_pairs)
    test_siamese_dataset = SiameseDataset(test_dataset, num_pairs=test_pairs)

    train_loader = DataLoader(train_siamese_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_siamese_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
