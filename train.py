import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import SiameseNetwork
from evaluate import evaluate_model

def train_model(train_loader, test_loader, device, num_epochs=10, lr=0.001, model_save_path=None):
    """训练孪生网络模型"""
    # 初始化模型、损失函数和优化器
    model = SiameseNetwork().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 用于记录训练过程
    train_losses = []
    test_losses = []
    accuracies = []

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for img1, img2, labels in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # 前向传播
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 测试阶段
        test_loss, accuracy = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    # 保存模型
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return model, {'train_losses': train_losses, 'test_losses': test_losses, 'accuracies': accuracies}
