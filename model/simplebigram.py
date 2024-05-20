'''
simple bigram model that has no implementation of attention
'''

import torch
import torch.nn as nn
from torch.functional import F

class SimpleBigram(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg['vocab_size']
        self.embedding_table = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, x, targets = None):
        logits = self.embedding_table(x)
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx