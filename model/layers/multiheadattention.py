'''
Takes the attention.py and creates a multiheadattention.py file
'''

import torch
import torch.nn as nn
from torch.functional import F

from model.layers.attention import Head

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads, n_embed, block_size, is_causal, use_rope=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, is_causal, use_rope) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size*num_heads, n_embed) # projection layer going back into the pathway
        self.dropout = nn.Dropout()

    def forward(self, x):
        B,T,C = x.shape
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.linear(out))
        return out