import torch
import torch.nn as nn
import torch.functional as F

import hydra

from trainers.dataloader import DataLoader
from trainers.utils import estimate_loss
from model import GPT2, SimpleBigram

from utils import save_checkpoint, load_checkpoint, ensure_dir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    
    ## dataloader
    data_loader = DataLoader(cfg = cfg)
    cfg['vocab_size'] = data_loader.tokenizer.vocab_size
    
    model_dict = {
        'simplebigram': SimpleBigram,
        'gpt2': GPT2
    }
    ## model
    model = model_dict[cfg.model.name](cfg = cfg).to(device)
    
    ## optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    ## ensure checkpoint directory
    checkpoint_dir = 'model/checkpoints'
    ensure_dir(checkpoint_dir)
    best_val_loss = float('inf')
    checkpoint_path = f'{checkpoint_dir}/{cfg.languagemodel.name}_model_checkpoint.pth'

    ## wandb
    if cfg.wandb_log:
        from utils import init_wandb
        init_wandb(cfg, model)

    ## train
    for iter in range(cfg.max_iters):
        
        if iter % cfg.eval_interval == 0:
            losses = estimate_loss(model = model,
                                   data_loader=data_loader,
                                   eval_iters=cfg.eval_iters)
            print(f'Iter {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')

            ## wandb
            if cfg.wandb_log:
                from utils import log_wandb
                log_wandb(losses)

            ## Save the model checkpoint if the validation loss is the best we have seen so far
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(model, optimizer, iter, best_val_loss, checkpoint_path)
        
        xb, yb = data_loader.load('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()