'''
Helper functions for the trainers objects
'''

import torch

@torch.no_grad()
def estimate_loss(model, data_loader, eval_iters):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x,y = data_loader.load(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out