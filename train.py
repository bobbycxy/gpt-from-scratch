import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
import torch.nn as nn
import torch.functional as F

import hydra

from trainers.dataloader import DataLoader
from trainers.utils import estimate_loss
from model import GPT2, SimpleBigram, build_optimizer

from utils import save_checkpoint, load_checkpoint, ensure_dir
from trainers.utils import build_dropout_scheduler, build_lr_scheduler
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    
    ## dataloader
    data_loader = DataLoader(cfg = cfg)
    cfg['vocab_size'] = data_loader.tokenizer.vocab_size
    
    ## model
    model_dict = {
        'simplebigram': SimpleBigram,
        'gpt2': GPT2
    }
    model = model_dict[cfg.model.name](cfg = cfg).to(device)
    
    ## optimizer
    optimizer = build_optimizer(model, 0.0, (0.9,0.999))

    ## scheduler
    # learning rate
    lr_scheduler = build_lr_scheduler(cfg)

    # dropout rate
    dropout_scheduler = build_dropout_scheduler(cfg)

    ## ensure checkpoint directory
    ensure_dir(cfg.model_ckpt_dir)
    best_val_loss = float('inf')
    checkpoint_path = f'{cfg.model_ckpt_dir}/{cfg.model_ckpt}'

    ## wandb
    if cfg.wandb_log:
        from utils import init_wandb
        init_wandb(cfg, model)

    ## train
    for iter in range(cfg.max_iters):
        
        # update learning rate and dropout rate
        lr_scheduler.step(optimizer)
        dropout_scheduler.step(model)
        
        if iter % cfg.eval_interval == 0:
            losses = estimate_loss(model = model,
                                   data_loader=data_loader,
                                   eval_iters=cfg.eval_iters)
            print(f'Iter {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')

            logs = {
                'epoch': iter,
                'train_loss': losses["train"],
                'val_loss': losses["val"],
                'lr': optimizer.param_groups[0]['lr'],
                'dropout_rate': np.mean([module.p for module in model.modules() if isinstance(module, nn.Dropout)])
            }

            ## wandb
            if cfg.wandb_log:
                from utils import log_wandb
                log_wandb(logs)

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