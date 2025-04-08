import torch
import torch.nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from models import SiameseNetwork
from evaluate import evaluate_model


def contrastive_loss(outputs, labels, margin=1.0):
    """
    对比损失函数:
    - 如果标签为1(相似对)，则最小化距离
    - 如果标签为0(不相似对)，则增大距离，直到达到边界(margin)
    """
    # outputs是相似度，转换回距离
    distances = -torch.log(torch.clamp(outputs, min=1e-8)) / 10.0  # 假设温度参数为10.0

    # 计算损失:
    # - 相似对(y=1): loss = distance^2
    # - 不相似对(y=0): loss = max(0, margin - distance)^2
    similar_loss = labels * torch.pow(distances, 2)
    dissimilar_loss = (1 - labels) * torch.pow(
        torch.clamp(margin - distances, min=0.0), 2
    )

    loss = torch.mean(similar_loss + dissimilar_loss)
    return loss


def train_model(
    train_loader,
    test_loader,
    device,
    num_epochs=10,
    lr=0.001,
    margin=1.0,
    model_save_path=None,
):
    """训练孪生网络模型"""
    # 初始化模型和优化器
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器 - 随着训练进行逐步降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 用于记录训练过程
    train_losses = []
    test_losses = []
    accuracies = []

    # 用于早停的变量
    best_accuracy = 0
    patience = 5
    patience_counter = 0

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for img1, img2, labels in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # 前向传播
            outputs = model(img1, img2).squeeze()
            loss = contrastive_loss(outputs, labels, margin)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算预测结果
            preds = (outputs > 0.5).float()
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 计算训练集准确率
        train_accuracy = np.mean(
            np.array(all_train_preds) == np.array(all_train_labels)
        )

        # 测试阶段
        test_loss, accuracy, _, _, _ = evaluate_model(
            model, test_loader, contrastive_loss, device
        )
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        # 学习率调度
        scheduler.step(test_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.4f}"
        )

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path} (accuracy: {accuracy:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # 如果启用了早停，加载最佳模型
    if patience_counter >= patience and model_save_path:
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded best model with accuracy: {best_accuracy:.4f}")

    return model, {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "accuracies": accuracies,
    }
