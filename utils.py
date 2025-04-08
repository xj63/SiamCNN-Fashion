import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import pandas as pd


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
    plt.figure(figsize=(16, 5))

    # 绘制损失
    plt.subplot(1, 3, 1)
    plt.plot(
        range(1, len(history["train_losses"]) + 1),
        history["train_losses"],
        "o-",
        label="Training Loss",
    )
    plt.plot(
        range(1, len(history["test_losses"]) + 1),
        history["test_losses"],
        "s-",
        label="Test Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.grid(True)

    # 绘制准确率
    plt.subplot(1, 3, 2)
    plt.plot(
        range(1, len(history["accuracies"]) + 1),
        history["accuracies"],
        "o-",
        label="Accuracy",
        color="green",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    plt.grid(True)

    # 绘制学习曲线 - 添加训练/验证集随样本量的变化
    if "learning_curve" in history:
        plt.subplot(1, 3, 3)
        learn_data = history["learning_curve"]
        plt.plot(
            learn_data["sizes"],
            learn_data["train_scores"],
            "o-",
            label="Training Score",
        )
        plt.plot(
            learn_data["sizes"],
            learn_data["test_scores"],
            "s-",
            label="Validation Score",
        )
        plt.xlabel("Training Size")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")


def visualize_similarity_distribution(similarities, labels, save_path=None):
    """可视化相似度分布"""
    plt.figure(figsize=(10, 6))

    # 分别获取相似对和不相似对的相似度值
    similar_scores = similarities[labels == 1]
    dissimilar_scores = similarities[labels == 0]

    # 绘制直方图
    plt.hist(similar_scores, bins=25, alpha=0.7, label="Similar Pairs", color="green")
    plt.hist(
        dissimilar_scores, bins=25, alpha=0.7, label="Dissimilar Pairs", color="red"
    )

    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Similarity Scores")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Similarity distribution plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Different", "Similar"],
        yticklabels=["Different", "Similar"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")


def visualize_tsne_embeddings(embeddings, labels, class_names, save_path=None):
    """使用t-SNE可视化特征嵌入空间"""
    # 降维到2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 创建颜色映射
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(12, 10))
    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            color=colors[i],
            label=class_names[label],
            alpha=0.7,
        )

    plt.title("t-SNE Visualization of Feature Embeddings")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")


def plot_difficult_pairs(
    original_data,
    pairs_indices,
    predictions,
    labels,
    similarity_scores,
    num_samples=5,
    save_path=None,
):
    """可视化模型误分类的困难样本对"""
    # 找出预测错误的样本
    errors = predictions != labels
    error_indices = np.where(errors)[0]

    if len(error_indices) == 0:
        print("No misclassified pairs found!")
        return

    # 选择错误样本中相似度最接近阈值的样本
    # 对于应该相似(1)但预测为不相似(0)的样本，相似度应该偏低
    fp_indices = error_indices[labels[error_indices] == 0]  # 假阳性
    fn_indices = error_indices[labels[error_indices] == 1]  # 假阴性

    # 按相似度排序
    if len(fp_indices) > 0:
        fp_indices = fp_indices[np.argsort(similarity_scores[fp_indices])][
            ::-1
        ]  # 从高到低
    if len(fn_indices) > 0:
        fn_indices = fn_indices[np.argsort(similarity_scores[fn_indices])]  # 从低到高

    # 选择样本
    num_fp = min(num_samples, len(fp_indices))
    num_fn = min(num_samples, len(fn_indices))

    plt.figure(figsize=(15, max(num_fp, num_fn) * 3))

    # 绘制假阳性样本（预测为相似但实际不相似）
    for i in range(num_fp):
        idx = fp_indices[i]
        pair_idx1, pair_idx2 = pairs_indices[idx]
        img1, _ = original_data[pair_idx1]
        img2, _ = original_data[pair_idx2]

        plt.subplot(max(num_fp, num_fn), 4, i * 4 + 1)
        plt.imshow(img1.squeeze().numpy(), cmap="gray")
        plt.title(f"FP Pair {i + 1} - Image 1")
        plt.axis("off")

        plt.subplot(max(num_fp, num_fn), 4, i * 4 + 2)
        plt.imshow(img2.squeeze().numpy(), cmap="gray")
        plt.title(f"Score: {similarity_scores[idx]:.3f}")
        plt.axis("off")

    # 绘制假阴性样本（预测为不相似但实际相似）
    for i in range(num_fn):
        idx = fn_indices[i]
        pair_idx1, pair_idx2 = pairs_indices[idx]
        img1, _ = original_data[pair_idx1]
        img2, _ = original_data[pair_idx2]

        plt.subplot(max(num_fp, num_fn), 4, i * 4 + 3)
        plt.imshow(img1.squeeze().numpy(), cmap="gray")
        plt.title(f"FN Pair {i + 1} - Image 1")
        plt.axis("off")

        plt.subplot(max(num_fp, num_fn), 4, i * 4 + 4)
        plt.imshow(img2.squeeze().numpy(), cmap="gray")
        plt.title(f"Score: {similarity_scores[idx]:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle(
        "Misclassified Pairs: False Positives (left) and False Negatives (right)",
        y=1.02,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Difficult pairs plot saved to {save_path}")


def plot_similarity_heatmap(similarity_matrix, class_names, save_path=None):
    """绘制类间相似度热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap="viridis",
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Inter-class Similarity Heatmap")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Similarity heatmap saved to {save_path}")


def export_analysis_report(
    predictions, labels, similarity_scores, threshold=0.5, file_path=None
):
    """导出详细分析报告"""
    # 创建预测标签
    pred_labels = (similarity_scores > threshold).astype(int)

    # 获取分类报告
    report = classification_report(
        labels, pred_labels, target_names=["Different", "Similar"], output_dict=True
    )

    # 转换为DataFrame
    df_report = pd.DataFrame(report).T

    # 相似度分析
    similarity_stats = {
        "true_similar": {
            "count": np.sum(labels == 1),
            "mean_sim": np.mean(similarity_scores[labels == 1]),
            "std_sim": np.std(similarity_scores[labels == 1]),
            "min_sim": np.min(similarity_scores[labels == 1]),
            "max_sim": np.max(similarity_scores[labels == 1]),
        },
        "true_different": {
            "count": np.sum(labels == 0),
            "mean_sim": np.mean(similarity_scores[labels == 0]),
            "std_sim": np.std(similarity_scores[labels == 0]),
            "min_sim": np.min(similarity_scores[labels == 0]),
            "max_sim": np.max(similarity_scores[labels == 0]),
        },
        "false_positives": {
            "count": np.sum((pred_labels == 1) & (labels == 0)),
            "mean_sim": np.mean(similarity_scores[(pred_labels == 1) & (labels == 0)])
            if np.any((pred_labels == 1) & (labels == 0))
            else 0,
            "min_sim": np.min(similarity_scores[(pred_labels == 1) & (labels == 0)])
            if np.any((pred_labels == 1) & (labels == 0))
            else 0,
            "max_sim": np.max(similarity_scores[(pred_labels == 1) & (labels == 0)])
            if np.any((pred_labels == 1) & (labels == 0))
            else 0,
        },
        "false_negatives": {
            "count": np.sum((pred_labels == 0) & (labels == 1)),
            "mean_sim": np.mean(similarity_scores[(pred_labels == 0) & (labels == 1)])
            if np.any((pred_labels == 0) & (labels == 1))
            else 0,
            "min_sim": np.min(similarity_scores[(pred_labels == 0) & (labels == 1)])
            if np.any((pred_labels == 0) & (labels == 1))
            else 0,
            "max_sim": np.max(similarity_scores[(pred_labels == 0) & (labels == 1)])
            if np.any((pred_labels == 0) & (labels == 1))
            else 0,
        },
    }

    # 转换为DataFrame
    df_sim_stats = pd.DataFrame.from_dict(
        {k: {k2: v2 for k2, v2 in v.items()} for k, v in similarity_stats.items()},
        orient="index",
    )

    # 如果提供了文件路径，保存到Excel
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with pd.ExcelWriter(file_path) as writer:
            df_report.to_excel(writer, sheet_name="Classification Report")
            df_sim_stats.to_excel(writer, sheet_name="Similarity Analysis")
        print(f"Analysis report saved to {file_path}")

    return df_report, df_sim_stats


def generate_class_examples(dataset, output_dir="./website/assets"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 类别名称
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # 为每个类别收集示例
    class_examples = defaultdict(list)

    # 获取每个类别的多个示例，然后选择最具代表性的一个
    for idx in range(min(5000, len(dataset))):
        img, label = dataset[idx]
        if len(class_examples[label]) < 20:  # 每个类别收集20个示例
            img_np = img.numpy().squeeze()  # 从tensor转换为numpy数组
            class_examples[label].append(img_np)

    # 为每个类别选择一个有代表性的示例并保存
    for label in range(10):
        examples = class_examples[label]
        if not examples:
            continue

        # 计算所有示例的平均图像
        mean_img = np.mean(examples, axis=0)

        # 找到与平均图像最接近的示例
        min_dist = float("inf")
        best_example = None
        for img in examples:
            dist = np.sum((img - mean_img) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_example = img

        # 保存这个示例
        plt.figure(figsize=(2, 2))
        plt.imshow(best_example, cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(
            f"{output_dir}/class_{label}.png",
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=150,
        )
        plt.close()
        print(f"Generated example image for class {label} ({class_names[label]})")

    print(f"All class example images have been saved to {output_dir}")


def generate_architecture_diagram(output_dir="./output"):
    """
    生成 Siamese Network 架构图

    参数:
        output_dir: 输出目录路径
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建架构图
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()

    # 设置背景为白色
    ax.set_facecolor("white")

    # 输入图像
    img1_pos = (1, 6)
    img2_pos = (1, 1)
    img_size = (2, 2)

    img1 = patches.Rectangle(
        img1_pos,
        *img_size,
        linewidth=1,
        edgecolor="black",
        facecolor="lightgray",
        label="Input Image 1",
    )
    img2 = patches.Rectangle(
        img2_pos,
        *img_size,
        linewidth=1,
        edgecolor="black",
        facecolor="lightgray",
        label="Input Image 2",
    )
    ax.add_patch(img1)
    ax.add_patch(img2)

    # 标注图像
    ax.text(
        img1_pos[0] + img_size[0] / 2,
        img1_pos[1] + img_size[1] / 2,
        "Image 1\n28x28x1",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        img2_pos[0] + img_size[0] / 2,
        img2_pos[1] + img_size[1] / 2,
        "Image 2\n28x28x1",
        ha="center",
        va="center",
        fontsize=10,
    )

    # CNN 特征提取器
    cnn1_pos = (4, 6)
    cnn2_pos = (4, 1)
    cnn_size = (2, 2)

    cnn1 = patches.Rectangle(
        cnn1_pos,
        *cnn_size,
        linewidth=1,
        edgecolor="black",
        facecolor="lightblue",
        label="CNN Feature Extractor",
    )
    cnn2 = patches.Rectangle(
        cnn2_pos,
        *cnn_size,
        linewidth=1,
        edgecolor="black",
        facecolor="lightblue",
        label="CNN Feature Extractor",
    )
    ax.add_patch(cnn1)
    ax.add_patch(cnn2)

    # 标注 CNN
    ax.text(
        cnn1_pos[0] + cnn_size[0] / 2,
        cnn1_pos[1] + cnn_size[1] / 2,
        "CNN\nFeature\nExtractor",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        cnn2_pos[0] + cnn_size[0] / 2,
        cnn2_pos[1] + cnn_size[1] / 2,
        "CNN\nFeature\nExtractor",
        ha="center",
        va="center",
        fontsize=10,
    )

    # 特征向量
    feat1_pos = (7, 6.25)
    feat2_pos = (7, 1.25)
    feat_size = (2, 1.5)

    feat1 = patches.Rectangle(
        feat1_pos,
        *feat_size,
        linewidth=1,
        edgecolor="black",
        facecolor="lightyellow",
        label="Feature Vector",
    )
    feat2 = patches.Rectangle(
        feat2_pos,
        *feat_size,
        linewidth=1,
        edgecolor="black",
        facecolor="lightyellow",
        label="Feature Vector",
    )
    ax.add_patch(feat1)
    ax.add_patch(feat2)

    # 标注特征向量
    ax.text(
        feat1_pos[0] + feat_size[0] / 2,
        feat1_pos[1] + feat_size[1] / 2,
        "Feature Vector\n64-dim",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        feat2_pos[0] + feat_size[0] / 2,
        feat2_pos[1] + feat_size[1] / 2,
        "Feature Vector\n64-dim",
        ha="center",
        va="center",
        fontsize=10,
    )

    # 距离计算
    dist_pos = (10, 3.5)
    dist_size = (2, 2)

    dist = patches.Rectangle(
        dist_pos,
        *dist_size,
        linewidth=1,
        edgecolor="black",
        facecolor="lightgreen",
        label="Distance Computation",
    )
    ax.add_patch(dist)

    # 标注距离计算
    ax.text(
        dist_pos[0] + dist_size[0] / 2,
        dist_pos[1] + dist_size[1] / 2,
        "Euclidean\nDistance",
        ha="center",
        va="center",
        fontsize=10,
    )

    # 输出
    out_pos = (13, 3.75)
    out_size = (1.5, 1.5)

    out = patches.Rectangle(
        out_pos,
        *out_size,
        linewidth=1,
        edgecolor="black",
        facecolor="lightcoral",
        label="Similarity Score",
    )
    ax.add_patch(out)

    # 标注输出
    ax.text(
        out_pos[0] + out_size[0] / 2,
        out_pos[1] + out_size[1] / 2,
        "Similarity\nScore\n(0-1)",
        ha="center",
        va="center",
        fontsize=10,
    )

    # 添加箭头连接
    # 图像到CNN
    ax.arrow(
        img1_pos[0] + img_size[0],
        img1_pos[1] + img_size[1] / 2,
        cnn1_pos[0] - img1_pos[0] - img_size[0] - 0.1,
        0,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )
    ax.arrow(
        img2_pos[0] + img_size[0],
        img2_pos[1] + img_size[1] / 2,
        cnn2_pos[0] - img2_pos[0] - img_size[0] - 0.1,
        0,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )

    # CNN到特征向量
    ax.arrow(
        cnn1_pos[0] + cnn_size[0],
        cnn1_pos[1] + cnn_size[1] / 2,
        feat1_pos[0] - cnn1_pos[0] - cnn_size[0] - 0.1,
        0,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )
    ax.arrow(
        cnn2_pos[0] + cnn_size[0],
        cnn2_pos[1] + cnn_size[1] / 2,
        feat2_pos[0] - cnn2_pos[0] - cnn_size[0] - 0.1,
        0,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )

    # 特征向量到距离计算
    ax.arrow(
        feat1_pos[0] + feat_size[0],
        feat1_pos[1] + feat_size[1] / 2,
        dist_pos[0] - feat1_pos[0] - feat_size[0] - 0.1,
        dist_pos[1] - feat1_pos[1] - feat_size[1] / 2 + dist_size[1] / 2,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )
    ax.arrow(
        feat2_pos[0] + feat_size[0],
        feat2_pos[1] + feat_size[1] / 2,
        dist_pos[0] - feat2_pos[0] - feat_size[0] - 0.1,
        dist_pos[1] + dist_size[1] / 2 - feat2_pos[1] - feat_size[1] / 2,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )

    # 距离计算到输出
    ax.arrow(
        dist_pos[0] + dist_size[0],
        dist_pos[1] + dist_size[1] / 2,
        out_pos[0] - dist_pos[0] - dist_size[0] - 0.1,
        0,
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )

    # 添加权重共享标识
    ax.plot([5, 5], [3, 4], "k--", linewidth=1)
    ax.text(
        5, 3.5, "Shared Weights", ha="center", va="center", fontsize=10, rotation=90
    )

    # 添加注释说明指数变换
    ax.text(
        12,
        5,
        "sim = exp(-dist)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
    )

    # 设置轴范围
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)

    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # 添加标题
    plt.title("Siamese Network Architecture with Euclidean Distance", fontsize=14)

    # 保存图片
    plt.savefig(f"{output_dir}/architecture.png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Architecture diagram has been saved to {output_dir}/architecture.png")
