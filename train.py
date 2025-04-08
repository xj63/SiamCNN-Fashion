import torch
import torch.nn
import torch.optim as optim
from tqdm import tqdm

from models import SiameseNetwork
from evaluate import evaluate_model


def train_model(
    train_loader,
    test_loader,
    device,
    num_epochs=10,
    lr=0.001,
    margin=1.0,
    model_save_path=None,
):
    """使用对比损失训练孪生网络模型"""
    # 初始化模型
    model = SiameseNetwork().to(device)

    # 使用对比损失(Contrastive Loss)代替BCELoss
    def contrastive_loss(outputs, labels, margin=margin):
        """
        对比损失函数:
        - 如果标签为1(相似对)，则最小化距离
        - 如果标签为0(不相似对)，则增大距离，直到达到边界(margin)
        """
        # outputs是相似度，转换回距离
        distances = -torch.log(outputs)

        # 计算损失:
        # - 相似对(y=1): loss = distance^2
        # - 不相似对(y=0): loss = max(0, margin - distance)^2
        similar_loss = labels * torch.pow(distances, 2)
        dissimilar_loss = (1 - labels) * torch.pow(
            torch.clamp(margin - distances, min=0.0), 2
        )

        loss = torch.mean(similar_loss + dissimilar_loss)
        return loss

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

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for img1, img2, labels in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # 前向传播
            outputs = model(img1, img2).squeeze()
            loss = contrastive_loss(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 测试阶段 - 使用修改后的评估函数
        test_loss, accuracy = evaluate_model(
            model, test_loader, contrastive_loss, device
        )
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

    # 保存模型
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    return model, {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "accuracies": accuracies,
    }
