'''
This is a script that reproduces Andrej Karparthy's 
code. It will be my attempt to para-code the learnings 
from scratch and test my ability to build an LLM from the ground up.

In the Bigram.py, we kept the character level language model simple. 
In this bigram-singleheadattention.py, we will add a single head attention
and observe how it's performance changes.

Results will be logged to wandb.
'''

## import libraries
import torch
import torch.nn as nn
from torch.functional import F

## ----- Hyperparameters ------
cfg = {
    'batch_size': 64, # B
    'block_size': 128, # T
    'n_embed': 64, # C
    'eval_iters': 10, # number of iterations to evaluate the model at a time step
    'max_iters': 3000, # number of iterations to train the model
    'eter_interval': 100, # interval to evaluate the model
    'learning_rate': 1e-3,
    'head_size': 8, # dimension of the head
    'num_heads': 8, # number of heads
    'num_layers': 4, # number of layers
    'dropout': 0.1 # dropout 
}

batch_size = cfg['batch_size']
block_size = cfg['block_size']
n_embed = cfg['n_embed']
eval_iters = cfg['eval_iters']
max_iters = cfg['max_iters']
eter_interval = cfg['eter_interval']
learning_rate = cfg['learning_rate']
head_size = cfg['head_size']
num_heads = cfg['num_heads']
num_layers = cfg['num_layers']
dropout = cfg['dropout']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'smallville'
wandb_run_name = 'multiheadattention-ffn-blocks-residconn-layernorm-dropout' # 'run' + str(time.time())


## 1. Import the data
with open('input.txt', 'r') as f:
    text = f.read()

## 2. Preprocess the data
char = sorted(list(set(text)))
vocab_size = len(char)

## 3. Create a mapping method
char_to_idx = {ch:i for i,ch in enumerate(char)}
idx_to_char = {i:ch for i,ch in enumerate(char)}
encode = lambda x: [char_to_idx[ch] for ch in x]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

## 4. Prepare the data
data = torch.tensor(encode(text), dtype=torch.long)
## Train and Val
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

## 5. Preparing the dataloader
## What is a dataloader? It is a function
## that will always return a batch of data that
## we can use to train the model.
def data_loader(split):
    '''
    Depending on the data set - e.g. train, validation - the function
    will return the data in batches. It starts with a batch of random
    integers and then returns the data in the batch size x block size.
    '''
    data = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx]) # B x T
    y = torch.stack([data[i+1:i+block_size+1] for i in idx]) # B x T
    return x.to(device), y.to(device)

## 6. Preparing the loss function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x,y = data_loader(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

## 7. Create the attention mechanism
class Head(nn.Module):
    def __init__(self, head_size):
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
    
## 8. Create the multi-head attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size*num_heads, n_embed) # projection layer going back into the pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.linear(out))
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed), # projection layer going back into the residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(head_size, num_heads)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x)) # pre-normalisation
        x = x + self.ffn(self.ln2(x)) # pre-normalisation
        return x

## 7. Define the model
class SuperSimpleBigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
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
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_trun = idx[:,-block_size:] # truncate the context
            logits, _ = self(idx_trun)
            logits = logits[:,-1,:] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
## 8. Initialize the model
model = SuperSimpleBigramModel(vocab_size).to(device)

## 9. Create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## wandb
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config = cfg)
    wandb.watch(model)

## 10. Train the model
for iter in range(max_iters):
    
    if iter % eter_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
        if wandb_log:
            wandb.log({'train_loss': losses['train'], 'val_loss': losses['val']})
    
    xb, yb = data_loader('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

## 11. Generate text
context = torch.zeros((1,1), dtype=torch.long).to(device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))