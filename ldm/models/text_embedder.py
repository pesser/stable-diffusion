import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.util import instantiate_from_config

class BaseTextEmbedder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, text_length, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.text_length = text_length
        self.word_embeddings = None
        self.tokenizer = None
        
        if 'init_method_std' in kwargs:
            init_method_std = kwargs.get('init_method_std', 0.02)
            torch.nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=init_method_std)
    
    def tokenize(self, x):
        raise NotImplementedError
    
    def detokenize(self, ids):
        raise NotImplementedError
    
    def encode(self, x, do_tokenize=False, **kwargs):
        if do_tokenize:
            ids = []
            for txt in x:
                ids.append(self.tokenize(txt))
            x = torch.tensor(ids)
            
        return self.word_embeddings(x).permute(0, 2, 1)
        
    def decode(self, x, do_detokenize=False, **kwargs): # x: b c l
        with torch.no_grad():
            x, word_embeddings_weight = x.cpu().detach(), self.word_embeddings.weight.cpu().detach()
            x = x[..., None].permute(0, 2, 3, 1) # bxlx1xc
            logits = (x - word_embeddings_weight).pow(2).mean(dim=-1)
            x = logits.argmax(dim=-1)
        
        if do_detokenize:
            txts = []
            for ids in x:
                txts.append(self.detokenize(ids))
            return txts
        else:
            return x

class RobertaTextEmbedder(BaseTextEmbedder):
    def __init__(self, ckpt_path, vocab_size, hidden_size, sample_length, tokenizer_path='', **kwargs):
        super().__init__(vocab_size, hidden_size, sample_length, **kwargs)
        state_dict = torch.load(ckpt_path)
        word_embeddings_weight = state_dict['module']['transformer.word_embeddings.weight']
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, 
                                                  _weight=word_embeddings_weight)
        
        from transformers import RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(os.path.join(tokenizer_path, 'roberta-large'), local_files_only=True)

    def tokenize(self, x):
        encoded_input = self.tokenizer(x, max_length=self.text_length, padding='max_length', truncation='only_first')
        return encoded_input['input_ids']
    
    def detokenize(self, ids):
        return self.tokenizer.decode(ids)
    