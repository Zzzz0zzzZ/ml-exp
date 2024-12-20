import torch
from transformers import BertTokenizer
from torch.utils.data import random_split
import pandas as pd
import numpy as np
import os

from config import get_args
from dataset import TextClassificationDataset
from model import BertClassifier
from trainer import Trainer


def main():
    args = get_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 打印超参数配置
    print("\n=== 模型配置 ===")
    print(f"模型名称: {args.model_name}")
    print(f"最大序列长度: {args.max_length}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.epochs}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"预热比例: {args.warmup_ratio}")
    print(f"早停耐心值: {args.early_stopping_patience}")
    print(f"随机种子: {args.seed}")
    print(f"设备: {args.device}")
    print("===============\n")

    # 检查数据文件是否存在
    for file_path in [args.train_file, args.test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到数据文件: {file_path}")

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # 准备数据集
    dataset = TextClassificationDataset(args.train_file, tokenizer, args.max_length)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    test_dataset = TextClassificationDataset(
        args.test_file,
        tokenizer,
        args.max_length,
        is_test=True
    )

    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size
    )

    # 初始化模型
    model = BertClassifier(args.model_name, args.num_classes)
    model = model.to(args.device)

    # 训练模型
    trainer = Trainer(model, args)
    best_val_acc = trainer.train(train_dataloader, val_dataloader)
    print(f'Best validation accuracy: {best_val_acc:.4f}')

    # 预测测试集
    predictions = trainer.predict(test_dataloader)

    print("\n=== Done. ===")


if __name__ == '__main__':
    main()
