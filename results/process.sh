#!/bin/bash

# 删除当前目录的submission.csv和submission.zip（如果存在的话）
rm -f submission.csv submission.zip

# 查找以submission_开头的csv文件，获取第一个匹配的文件（如果有）并重命名为submission.csv
submission_file=$(find . -maxdepth 1 -type f -name "submission_*.csv" | head -n 1)
if [ -n "$submission_file" ]; then
    mv "$submission_file" submission.csv
fi

# 将重命名后的submission.csv文件压缩成submission.zip
if [ -f "submission.csv" ]; then
    zip submission.zip submission.csv
fi