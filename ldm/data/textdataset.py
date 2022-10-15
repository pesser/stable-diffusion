import os
import numpy as np
from torch.utils.data import Dataset
import csv

# add tokenizer
pretrain_path=''
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'), local_files_only=True)

def _encode(text, sample_length):
    encoded_input = tokenizer(text, max_length=sample_length, padding='max_length', truncation='only_first')
    return encoded_input['input_ids']

class E2EDataset(Dataset):
    def __init__(self,
                 path, sample_length):
        self.sample_length = sample_length
        items = []
        with open(path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            
            for line in reader:
                items.append({key: line[i] for i, key in enumerate(header)})
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        data = self.items[idx]
        data['input_ids'] = np.array(_encode(data['ref'], self.sample_length), dtype=np.int64)
        return data
        