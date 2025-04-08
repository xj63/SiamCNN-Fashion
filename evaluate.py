import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import (
    plot_confusion_matrix,
    visualize_similarity_distribution,
    visualize_tsne_embeddings,
    plot_difficult_pairs,
    plot_similarity_heatmap,
    export_analysis_report,
)
import os


def evaluate_model(model, test_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算预测结果 - 基于相似度阈值
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return (
        test_loss,
        accuracy,
        np.array(all_scores),
        np.array(all_labels),
        np.array(all_preds),
    )


def extract_all_features(model, data_loader, device):
    """提取数据集中所有样本的特征"""
    model.eval()
    all_features = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(
            tqdm(data_loader, desc="Extracting features")
        ):
            images = images.to(device)
            features = model.feature_extractor(images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_indices.extend(
                range(
                    batch_idx * data_loader.batch_size,
                    min(
                        (batch_idx + 1) * data_loader.batch_size,
                        len(data_loader.dataset),
                    ),
                )
            )

    return np.vstack(all_features), np.concatenate(all_labels), np.array(all_indices)


def compute_class_similarity_matrix(model, data_loader, device, num_classes=10):
    """计算类间相似度矩阵"""
    # 提取所有样本的特征
    features, labels, _ = extract_all_features(model, data_loader, device)

    # 初始化相似度矩阵
    similarity_matrix = np.zeros((num_classes, num_classes))

    # 对每一类计算类间相似度
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                # 同类相似度设为1.0
                similarity_matrix[i, j] = 1.0
            else:
                # 不同类间计算平均相似度
                class_i_features = features[labels == i]
                class_j_features = features[labels == j]

                # 计算每对样本间相似度，取样避免计算量过大
                if len(class_i_features) > 20:
                    class_i_features = class_i_features[
                        np.random.choice(len(class_i_features), 20, replace=False)
                    ]
                if len(class_j_features) > 20:
                    class_j_features = class_j_features[
                        np.random.choice(len(class_j_features), 20, replace=False)
                    ]

                sim_scores = []
                for feat_i in class_i_features:
                    for feat_j in class_j_features:
                        # 计算欧氏距离并转换为相似度
                        dist = np.sqrt(np.sum((feat_i - feat_j) ** 2))
                        sim = np.exp(-dist)
                        sim_scores.append(sim)

                # 取平均值
                similarity_matrix[i, j] = np.mean(sim_scores)

    return similarity_matrix


def detailed_evaluation(
    model,
    test_loader,
    device,
    original_dataset=None,
    class_names=None,
    output_dir="./analysis",
):
    """详细评估模型，生成可视化和分析报告"""
    print("Performing detailed evaluation...")
    model.eval()

    all_similarities = []
    all_labels = []
    all_features1 = []
    all_features2 = []
    all_pair_indices = []

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for img1, img2, labels in tqdm(test_loader, desc="Evaluating"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # 提取特征向量
            feat1 = model.forward_one(img1)
            feat2 = model.forward_one(img2)

            # 计算相似度
            outputs = model(img1, img2).squeeze()

            all_similarities.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_features1.extend(feat1.cpu().numpy())
            all_features2.extend(feat2.cpu().numpy())

            # 如果测试加载器包含对索引，也保存它们
            if hasattr(test_loader.dataset, "pairs"):
                batch_start = len(all_labels) - len(labels)
                for i in range(len(labels)):
                    all_pair_indices.append(
                        test_loader.dataset.pairs[batch_start + i][0]
                    )

    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    all_preds = (all_similarities > 0.5).astype(int)

    # 1. 计算基本指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 2. ROC曲线和AUC
    fpr, tpr, _ = roc_curve(all_labels, all_similarities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()

    # 3. 相似度分布可视化
    visualize_similarity_distribution(
        all_similarities,
        all_labels,
        save_path=f"{output_dir}/similarity_distribution.png",
    )

    # 4. 混淆矩阵
    plot_confusion_matrix(
        all_labels,
        all_preds,
        ["Different", "Similar"],
        save_path=f"{output_dir}/confusion_matrix.png",
    )

    # 5. 导出分析报告
    export_analysis_report(
        all_preds,
        all_labels,
        all_similarities,
        file_path=f"{output_dir}/analysis_report.xlsx",
    )

    # 6. 如果提供了原始数据集，绘制困难样本
    if original_dataset is not None and len(all_pair_indices) > 0:
        plot_difficult_pairs(
            original_dataset,
            all_pair_indices,
            all_preds,
            all_labels,
            all_similarities,
            num_samples=8,
            save_path=f"{output_dir}/difficult_pairs.png",
        )

    # 7. 如果提供了类名，计算并绘制类间相似度矩阵
    if class_names is not None and hasattr(test_loader.dataset, "dataset"):
        # 创建单样本数据加载器
        single_loader = torch.utils.data.DataLoader(
            test_loader.dataset.dataset, batch_size=64, shuffle=False
        )

        sim_matrix = compute_class_similarity_matrix(
            model, single_loader, device, len(class_names)
        )
        plot_similarity_heatmap(
            sim_matrix,
            class_names,
            save_path=f"{output_dir}/class_similarity_matrix.png",
        )

    # 8. 使用t-SNE可视化嵌入空间
    # 需要单样本特征和标签
    if hasattr(test_loader.dataset, "dataset"):
        # 处理大型数据集 - 随机采样以避免过长处理时间
        all_feats = np.vstack([all_features1, all_features2])
        indices = np.random.choice(
            len(all_feats), min(2000, len(all_feats)), replace=False
        )

        # 假设原始数据集标签可以从test_loader的dataset属性中获取
        single_loader = torch.utils.data.DataLoader(
            test_loader.dataset.dataset, batch_size=64, shuffle=False
        )
        features, labels, _ = extract_all_features(model, single_loader, device)

        if len(features) > 3000:
            # 随机采样以避免过长处理时间
            indices = np.random.choice(len(features), 3000, replace=False)
            features = features[indices]
            labels = labels[indices]

        if class_names:
            visualize_tsne_embeddings(
                features,
                labels,
                class_names,
                save_path=f"{output_dir}/tsne_embeddings.png",
            )

    print(f"Detailed evaluation complete. Results saved to {output_dir}")
    return accuracy
