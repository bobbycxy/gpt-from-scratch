'''
Network to house the GPT2 model
'''

import torch
import torch.nn as nn
from torch.functional import F

from model.blocks.gpt2decoder import DecoderBlock as Block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg['vocab_size']
        self.num_layers = cfg['num_layers']
        self.num_heads = cfg['num_heads']
        self.n_embed = cfg['n_embed']
        self.block_size = cfg['block_size']
        self.dropout = cfg['dropout']
        self.is_causal = cfg['is_causal']

        self.embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)
        self.blocks = nn.Sequential(*[Block(self.num_heads, self.n_embed, self.block_size, self.dropout, self.is_causal) for _ in range(self.num_layers)])
        self.ln_final = nn.LayerNorm(self.n_embed)
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size)

    def forward(self, idx, targets = None):
        B,T = idx.shape
        tok_emb = self.embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T).to(device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index = -100) ## ignore the padding tokens
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_trun = idx[:,-self.block_size:] # truncate the context
            logits, _ = self(idx_trun)
            logits = logits[:,-1,:] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx