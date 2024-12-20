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

    # 保存预测结果
    submission = pd.DataFrame({
        'index': range(len(predictions)),
        'label': predictions
    })

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    submission.to_csv(args.output_file, index=False)
    print(f'预测结果已保存到: {args.output_file}')


if __name__ == '__main__':
    main()
