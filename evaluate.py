import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def evaluate_model(model, test_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算预测结果
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return test_loss, accuracy

def detailed_evaluation(model, test_loader, device):
    """详细评估模型，包括准确率和其他指标"""
    model.eval()
    all_preds = []
    all_labels = []

    print("Performing detailed evaluation...")
    with torch.no_grad():
        for img1, img2, labels in tqdm(test_loader, desc="Evaluating"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            outputs = model(img1, img2).squeeze()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    # 这里可以添加更多评估指标，如精确率、召回率、F1分数等

    print(f'Final Test Accuracy: {accuracy:.4f}')

    return accuracy
