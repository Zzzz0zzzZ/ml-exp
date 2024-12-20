import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_root, 'datasets')

    # 常用
    parser.add_argument('--alias', type=str, default='bert-base-uncased', help='实验的别名，用于保存权重和结果文件')
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--train_ratio', type=float, default=0.9, help='训练集占总数据集的比例')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2e-5)

    # 不常用
    parser.add_argument('--train_file', type=str, default=os.path.join(dataset_dir, 'train.txt'))
    parser.add_argument('--test_file', type=str, default=os.path.join(dataset_dir, 'test.txt'))
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # 确保数据目录存在
    os.makedirs(dataset_dir, exist_ok=True)

    return args
