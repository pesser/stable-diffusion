import os
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.util import instantiate_from_config
    

class RobertaTextEmbedder(nn.Module):
    def __init__(self, vocab_size, hidden_size, sample_length, embedding_ckpt_path=None, embedding_init_std=0.02, tokenizer_path='', **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.text_length = sample_length
        
        if embedding_ckpt_path is not None:
            state_dict = torch.load(embedding_ckpt_path)
            word_embeddings_weight = state_dict['module']['transformer.word_embeddings.weight']
            self.word_embeddings = nn.Embedding(vocab_size, hidden_size, 
                                                  _weight=word_embeddings_weight)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        from transformers import RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(os.path.join(tokenizer_path, 'roberta-large'), local_files_only=True)
    
    def tokenize(self, x):
        encoded_input = self.tokenizer(x, max_length=self.text_length, padding='max_length', truncation='only_first')
        return encoded_input['input_ids']

    def detokenize(self, ids):
        return self.tokenizer.decode(ids)
    
    def encode(self, x, do_tokenize=False, **kwargs):
        if do_tokenize:
            ids = []
            for txt in x:
                ids.append(self.tokenize(txt))
            x = torch.tensor(ids).cuda()
            
        return self.word_embeddings(x).permute(0, 2, 1)
    
    def decode(self, x, do_detokenize=False, **kwargs): # x: b c l
        with torch.no_grad():
            x, word_embeddings_weight = x.detach().cpu(), self.word_embeddings.weight.detach().cpu()
            x = x[..., None].permute(0, 2, 3, 1) # bxlx1xc
            logits = (x - word_embeddings_weight).pow(2).mean(dim=-1)
            x = logits.argmin(dim=-1)
        
        if do_detokenize:
            txts = []
            for ids in x:
                txts.append(self.detokenize(ids))
            return txts
        else:
            return x

class RobertaTextEmbedderLMHead(RobertaTextEmbedder):
    def __init__(self, vocab_size, hidden_size, sample_length, embedding_ckpt_path=None, embedding_init_std=None, tokenizer_path='', **kwargs):
        super().__init__(vocab_size, hidden_size, sample_length, embedding_ckpt_path, embedding_init_std, tokenizer_path, **kwargs)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def decode(self, x, do_detokenize=False, **kwargs): # x: b c l
        logits = self.get_logits(x)
        x = logits.argmax(dim=-1)
        
        if do_detokenize:
            txts = []
            for ids in x:
                txts.append(self.detokenize(ids))
            return txts
        else:
            return x
    
    def get_logits(self, x):
        x = x.permute(0, 2, 1)
        return self.lm_head(x)