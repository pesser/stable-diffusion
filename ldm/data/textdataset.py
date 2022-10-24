import os
import numpy as np
from torch.utils.data import Dataset
import csv

from ldm.util import tokenize

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
        data['input_ids'] = np.array(tokenize(data['ref'], self.sample_length), dtype=np.int64)
        return data
        
class QQPDataset(Dataset):
    def __init__(self,
                 path, sample_length, cond_length):
        self.sample_length = sample_length
        self.cond_length = cond_length
        
        data = []
        with open(path, 'r') as f:
            head = f.readline()
            for line in f:
                try:
                    items = line.split('\t')
                    if int(items[-1]) == 0:
                        continue
                    data.append((items[3], items[4]))
                except:
                    continue
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        data = {}
        
        data['cond_input_ids'] = np.array(tokenize(item[0], self.sample_length), dtype=np.int64)
        data['input_ids'] = np.array(tokenize(item[1], self.cond_length), dtype=np.int64)
        return data
        
                
        