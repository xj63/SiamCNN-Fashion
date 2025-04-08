import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """CNN特征提取器"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 64)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    """孪生网络模型"""
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        """对单个输入进行特征提取"""
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        """对两个输入进行处理并计算相似度"""
        # 提取两个输入的特征
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)

        # 拼接特征
        combined = torch.cat((feat1, feat2), 1)

        # 计算相似度
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return self.sigmoid(x)
