import os
import argparse
from data import load_fashion_mnist, create_data_loaders
from train import train_model
from evaluate import detailed_evaluation
from utils import set_seed, get_device, plot_training_history

def main(args):
    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device()
    print(f"Using device: {device}")

    # 加载数据
    print("Loading Fashion MNIST dataset...")
    train_dataset, test_dataset = load_fashion_mnist(root=args.data_dir)

    # 创建数据加载器
    print("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=args.batch_size,
        train_pairs=args.train_pairs,
        test_pairs=args.test_pairs
    )

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, 'siamese_model.pth')
    plot_save_path = os.path.join(args.output_dir, 'training_history.png')

    # 训练模型
    print("Training model...")
    model, history = train_model(
        train_loader,
        test_loader,
        device,
        num_epochs=args.epochs,
        lr=args.learning_rate,
        model_save_path=model_save_path
    )

    # 绘制训练历史
    plot_training_history(history, save_path=plot_save_path)

    # 详细评估
    print("\nPerforming detailed evaluation...")
    detailed_evaluation(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate siamese CNN for Fashion MNIST")

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store datasets')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save outputs')
    parser.add_argument('--train_pairs', type=int, default=10000, help='Number of training pairs')
    parser.add_argument('--test_pairs', type=int, default=2000, help='Number of test pairs')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)
