'''
Create a single attention head
'''

import torch
import torch.nn as nn
from torch.functional import F

class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.q = nn.Linear(n_embed, head_size)
        self.k = nn.Linear(n_embed, head_size)
        self.v = nn.Linear(n_embed, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # used to mask the attention matrix
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        ## Scaled dot product attention
        attention = Q @ K.transpose(-2,-1) * (C ** -0.5) # B,T,T
        attention = attention.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # B,T,T
        attention = F.softmax(attention, dim=-1) # B,T,T
        attention = self.dropout(attention)
        out = attention @ V # B,T,H
        return out
