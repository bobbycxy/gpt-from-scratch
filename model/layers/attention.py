'''
Create a single attention head
'''

import torch
import torch.nn as nn
from torch.functional import F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Head(nn.Module):
    def __init__(self, head_size, n_embed, block_size, is_causal, use_rope=False):
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

        self.use_rope = use_rope

    def forward(self, x, mask=None):
        B,T,C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        ## If we are using the rope embeddings
        if self.use_rope:
            ## prep the freqs_cis
            dim = Q.shape[-1]
            end = Q.shape[1]
            freqs_cis = precompute_freqs_cis(dim, end)

            ## get the rope embeddings of Q and K
            Q, K = apply_rotary_emb(Q, K, freqs_cis)

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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(device)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # Reshape to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast frequencies
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply the rotary embeddings
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(*xq.shape)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(*xk.shape)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

## Commenting this because this is exactly the same as Llama's implementation.
## However, the model uses 3D dimensions si the implementation is slightly different.
## I will keep this here for reference.
# def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)