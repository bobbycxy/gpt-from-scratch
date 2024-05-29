'''
Create a single attention head
'''

import torch
import torch.nn as nn
from torch.functional import F

class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size, is_causal):
        super().__init__()
        self.head_size = head_size
        self.q = nn.Linear(n_embed, head_size)
        self.k = nn.Linear(n_embed, head_size)
        self.v = nn.Linear(n_embed, head_size)
        if is_causal:
            self.register_buffer('triu', torch.triu(torch.ones(block_size, block_size), diagonal=1)) # If causal, then the upper half of the matrix is masked. This would be causal self-attention
        else:
            self.register_buffer('triu', torch.triu(torch.zeros(block_size, block_size), diagonal=1)) # if non-causal, then the whole matrix is not masked. This would be bidirectional self-attention
        self.dropout = nn.Dropout()

    def forward(self, x, mask=None):
        B,T,C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        ## If there's a padding mask, we want to adjust the attention mask.
        if mask is not None: # if there's a padding mask to factor in input tokens that have the pad_token_id
            causal_mask = self.triu[:T,:T].unsqueeze(0) == 1 # upper half of the matrix is True
            ## mask is True where the padding token is
            mask = (mask == 0) ## ! - Need to replace 0 with self.tokenizer.pad_token_id
            attn_mask = causal_mask + mask.unsqueeze(-1) # True + False = True
        else: 
            attn_mask = self.triu[:T,:T].unsqueeze(0) == 1 # upper half of the matrix is True

        ## Scaled dot product attention
        attention = Q @ K.transpose(-2,-1) * (C ** -0.5) # B,T,T
        attention = attention.masked_fill(attn_mask, float('-inf')) # B,T,T # replaces the True values with -inf
        attention = F.softmax(attention, dim=-1) # B,T,T
        attention = self.dropout(attention)
        out = attention @ V # B,T,H
        return out
