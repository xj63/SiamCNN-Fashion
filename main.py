import os
import argparse
import torch
from data import load_fashion_mnist, create_data_loaders, FASHION_MNIST_CLASSES
from train import train_model
from evaluate import detailed_evaluation
from utils import (
    set_seed,
    get_device,
    plot_training_history,
    generate_class_examples,
    generate_architecture_diagram,
)


def main(args):
    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device()
    print(f"Using device: {device}")

    # 加载数据
    print("Loading Fashion MNIST dataset...")
    train_dataset, test_dataset = load_fashion_mnist(root=args.data_dir)

    # 创建类别示例图片
    class_img_dir = os.path.join(args.output_dir, "example_img")
    if not os.path.exists(class_img_dir) or not all(
        os.path.exists(os.path.join(class_img_dir, f"class_{i}.png")) for i in range(10)
    ):
        print("Generating class example images...")
        generate_class_examples(train_dataset, output_dir=class_img_dir)

    # 生成架构图
    architecture_path = args.output_dir
    print("Generating architecture diagram for website...")
    generate_architecture_diagram(output_dir=architecture_path)

    # 创建数据加载器
    print("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=args.batch_size,
        train_pairs=args.train_pairs,
        test_pairs=args.test_pairs,
    )

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, "siamese_model.pth")
    plot_save_path = os.path.join(args.output_dir, "training_history.png")
    analysis_dir = os.path.join(args.output_dir, "analysis")

    # 训练模型
    if not args.skip_training:
        print("Training model...")
        model, history = train_model(
            train_loader,
            test_loader,
            device,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            margin=args.margin,
            model_save_path=model_save_path,
        )

        # 绘制训练历史
        plot_training_history(history, save_path=plot_save_path)
    else:
        # 加载已有模型
        print(f"Loading model from {model_save_path}...")
        from models import SiameseNetwork

        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    # 详细评估
    print("\nPerforming detailed evaluation...")
    detailed_evaluation(
        model,
        test_loader,
        device,
        original_dataset=test_dataset,  # 提供原始数据集以绘制困难样本
        class_names=FASHION_MNIST_CLASSES,  # 提供类别名称
        output_dir=analysis_dir,  # 分析结果保存路径
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate siamese CNN for Fashion MNIST"
    )

    # 数据参数
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory to store datasets"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--train_pairs", type=int, default=10000, help="Number of training pairs"
    )
    parser.add_argument(
        "--test_pairs", type=int, default=2000, help="Number of test pairs"
    )

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--margin", type=float, default=1.0, help="Margin for contrastive loss"
    )

    # 添加跳过训练的选项（用于直接评估已有模型）
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and load existing model",
    )

    args = parser.parse_args()
    main(args)
