'''
This is a script that reproduces Andrej Karparthy's 
code. It will be my attempt to para-code the learnings 
from scratch and test my ability to build an LLM from the ground up.
'''

## import libraries
import torch
import torch.nn as nn
from torch.functional import F

## ----- Hyperparameters ------
cfg = {
    'batch_size': 64,
    'block_size': 128,
    'eval_iters': 10,
    'max_iters': 3000,
    'eter_interval': 100,
    'learning_rate': 1e-3
}

batch_size = cfg['batch_size']
block_size = cfg['block_size']
eval_iters = cfg['eval_iters']
max_iters = cfg['max_iters']
eter_interval = cfg['eter_interval']
learning_rate = cfg['learning_rate']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb logging
wandb_log = True # disabled by default
wandb_project = 'smallville'
wandb_run_name = 'simplebigram' # 'run' + str(time.time())


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

## 7. Define the model
class SuperSimpleBigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets = None):
        logits = self.embedding(idx)

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
            logits, _ = self(idx)
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