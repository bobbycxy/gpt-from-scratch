import torch
import torch.nn as nn
import torch.functional as F

import hydra

from trainers.dataloader import DataLoader
from trainers.utils import estimate_loss
from model import GPT2, SimpleBigram

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    
    ## dataloader
    data_loader = DataLoader(cfg = cfg)
    
    model_dict = {
        'simplebigram': SimpleBigram,
        'gpt2': GPT2
    }
    ## model
    model = model_dict[cfg.model.name](cfg = cfg).to(device)
    
    ## optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    ## train
    for iter in range(cfg.max_iters):
        
        if iter % cfg.eval_interval == 0:
            losses = estimate_loss(model = model,
                                   data_loader=data_loader,
                                   eval_iters=cfg.eval_iters)
            print(f'Iter {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
        
        xb, yb = data_loader.load('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()