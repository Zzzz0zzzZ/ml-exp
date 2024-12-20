# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def analyze_lengths(file_path):
    lengths = []

    # 读取文件并统计长度
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉标签和分隔符,只保留文本内容
            text = line.strip().split('+++$+++')[1].strip()
            # 统计词数(以空格分割)
            length = len(text.split())
            lengths.append(length)

    # 计算统计指标
    max_len = max(lengths)
    min_len = min(lengths)
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)

    # 计算分位数
    percentiles = np.percentile(lengths, [75, 90, 95, 99])

    # 打印统计信息
    print(f'数据集句子长度统计:')
    print(f'最短长度: {min_len}')
    print(f'最长长度: {max_len}')
    print(f'平均长度: {mean_len:.2f}')
    print(f'中位数长度: {median_len}')
    print(f'75%分位数: {percentiles[0]}')
    print(f'90%分位数: {percentiles[1]}')
    print(f'95%分位数: {percentiles[2]}')
    print(f'99%分位数: {percentiles[3]}')

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('句子长度分布')
    plt.xlabel('句子长度(词数)')
    plt.ylabel('频数')

    # 添加垂直线标注重要分位数
    plt.axvline(percentiles[1], color='r', linestyle='--', label='90%分位数')
    plt.axvline(percentiles[2], color='g', linestyle='--', label='95%分位数')
    plt.axvline(percentiles[3], color='y', linestyle='--', label='99%分位数')

    plt.legend()
    plt.show()


# 分析数据集
analyze_lengths('datasets/train.txt')