import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

# Assuming the CSV has two columns: 'tweet' and 'label'
class TweetsDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len):
        self.data = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet = self.data.loc[index, 'tweet']
        label = self.data.loc[index, 'label']
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


