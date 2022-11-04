import os
import json
import numpy as np
from torch.utils.data import Dataset
import csv

from ldm.util import tokenize

class E2EDataset(Dataset):
    def __init__(self,
                 path, sample_length):
        super().__init__()
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
                 path, sample_length, cond_length, num_shards=1, shard_id=0):
        super().__init__()
        self.sample_length = sample_length
        self.cond_length = cond_length
        
        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            head = next(reader)
            if 'test' in path:
                for items in reader:
                    try:
                        data.append((items[1], items[2]))
                    except:
                        continue
            else:
                for items in reader:
                    try:
                        # items = line.split(',')
                        if int(items[-1]) == 0:
                            continue
                        data.append((items[3], items[4]))
                    except:
                        continue

        length_shard = len(data) // num_shards
        if shard_id == num_shards - 1:
            self.data = data[shard_id*length_shard:]
        else:
            self.data = data[shard_id*length_shard:(shard_id+1)*length_shard]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        data = {
            "cond_input_ids": np.array(tokenize(item[0], self.sample_length), dtype=np.int64),
            "input_ids": np.array(tokenize(item[1], self.cond_length), dtype=np.int64)
        }
        return data
        
class QuasarTDataset(Dataset):
    def __init__(self,
                 path, sample_length, cond_length, num_shards=1, shard_id=0):
        super().__init__()
        self.sample_length = sample_length
        self.cond_length = cond_length
        
        answer_path = path.split('.')[0] + '.txt'
        answers = []
        with open(answer_path, 'r') as answer_f:
            for line in answer_f:
                answers.append(json.loads(line)["answers"][0])
        
        data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                for dic in item:
                    document = ' '.join(dic["document"])
                    if not answers[i] in document:
                        continue
                    question = ' '.join(dic["question"])
                    data.append((document, question))
        
        length_shard = len(data) // num_shards
        if shard_id == num_shards - 1:
            self.data = data[shard_id*length_shard:]
        else:
            self.data = data[shard_id*length_shard:(shard_id+1)*length_shard]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        data = {
            "cond_input_ids": np.array(tokenize(item[0], self.sample_length), dtype=np.int64),
            "input_ids": np.array(tokenize(item[1], self.cond_length), dtype=np.int64)
        }
        return data
        
class CCDataset(Dataset):
    def __init__(self, 
                 path, sample_length, cond_length, num_shards=1, shard_id=0):
        super().__init__()
        self.sample_length = sample_length
        self.cond_length = cond_length
        
        data = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                post = ' '.join(item["post"])
                response = ' '.join(item["response"])
                data.append((post, response))
        
        length_shard = len(data) // num_shards
        if shard_id == num_shards - 1:
            self.data = data[shard_id*length_shard:]
        else:
            self.data = data[shard_id*length_shard:(shard_id+1)*length_shard]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        data = {
            "cond_input_ids": np.array(tokenize(item[0], self.sample_length), dtype=np.int64),
            "input_ids": np.array(tokenize(item[1], self.cond_length), dtype=np.int64)
        }
        return data
        