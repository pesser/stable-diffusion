import os
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.util import instantiate_from_config
from ldm.util import get_tokenizer, tokenize, detokenize
    

class TextEmbedder(nn.Module):
    def __init__(self, vocab_size, hidden_size, sample_length, embedding_ckpt_path=None, embedding_init_std=0.02, **kwargs):
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
    
    def encode(self, x, **kwargs):            
        return self.word_embeddings(x).permute(0, 2, 1)
    
    def decode(self, x, **kwargs): # x: b c l
        with torch.no_grad():
            x, word_embedding_weight = x.permute(0, 2, 1).detach(), self.word_embeddings.weight.detach()
            x_square, word_embedding_weight_square = x.pow(2).mean(dim=-1), word_embedding_weight.pow(2).mean(dim=-1)
            x_mult_word_embedding_weight = x @ word_embedding_weight.permute(1, 0) / self.hidden_size
            logits = x_square.unsqueeze(-1) - 2 * x_mult_word_embedding_weight + word_embedding_weight_square
            
            # x, word_embeddings_weight = x.detach().cpu(), self.word_embeddings.weight.detach().cpu()
            # x = x[..., None].permute(0, 2, 3, 1) # bxlx1xc
            # logits_2 = (x - word_embeddings_weight).pow(2).mean(dim=-1)
        return logits.argmin(dim=-1)

    def forward(self, x):
        x = x.to(self.device)
        return self.encode(x)

class TextEmbedderLMHead(TextEmbedder):
    def __init__(self, vocab_size, hidden_size, sample_length, embedding_ckpt_path=None, embedding_init_std=0.02, **kwargs):
        super().__init__(vocab_size, hidden_size, sample_length, embedding_ckpt_path, embedding_init_std, tokenizer_path, **kwargs)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def decode(self, x, do_detokenize=False, **kwargs): # x: b c l
        logits = self.get_logits(x)
        x = logits.argmax(dim=-1)
        return x
    
    def get_logits(self, x):
        x = x.permute(0, 2, 1)
        return self.lm_head(x)