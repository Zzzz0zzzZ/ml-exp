import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, is_test=False):
        self.data = []
        self.labels = []
        self.is_test = is_test

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if is_test:
                    idx, text = line.strip().split(',', 1)
                    self.data.append(text)
                else:
                    label, text = line.strip().split('+++$+++')
                    self.data.append(text.strip())
                    self.labels.append(int(label))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

        if not self.is_test:
            item['label'] = torch.tensor(self.labels[idx])

        return item
