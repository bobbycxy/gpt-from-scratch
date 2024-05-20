'''
Decoder block for the GPT model
'''

import torch
import torch.nn as nn
from torch.functional import F

from model.layers import MultiHeadAttention, FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, num_heads, n_embed, block_size, dropout, is_causal):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(head_size, num_heads, n_embed, block_size, dropout, is_causal)
        self.ffn = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x)) # pre-normalisation
        x = x + self.ffn(self.ln2(x)) # pre-normalisation
        return x