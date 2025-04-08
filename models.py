import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32 * 7 * 7, 64)
        self.bn3 = nn.BatchNorm1d(64)  # 对最终特征进行标准化，有助于欧氏距离计算

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)

        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc(x)))

        # L2正则化特征向量，使其在单位超球面上
        x = F.normalize(x, p=2, dim=1)

        return x


class SiameseNetwork(nn.Module):
    """基于欧氏距离的孪生网络模型"""

    def __init__(self, temperature=1.0):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.temperature = temperature  # 用于缩放距离的温度参数

    def forward_one(self, x):
        """对单个输入进行特征提取"""
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        """对两个输入进行处理并计算相似度"""
        # 提取特征向量
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)

        # 计算欧氏距离
        dist = torch.sqrt(torch.sum((feat1 - feat2) ** 2, dim=1, keepdim=True))

        # 使用负指数函数将距离转换为相似度分数 (0到1之间)
        # 添加温度参数以调整相似度曲线的陡峭程度
        similarity = torch.exp(-self.temperature * dist)

        return similarity
