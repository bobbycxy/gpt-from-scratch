import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
import torch.nn as nn
import torch.functional as F

import hydra

from trainers.dataloader import DataLoader
from trainers.utils import estimate_loss
from model import GPT2, SimpleBigram

from utils import save_checkpoint, load_checkpoint, ensure_dir
from trainers.scheduler import *
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
    param_dict = {pn: p for pn, p in model.named_parameters()} ## get each layer of the model
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} ## get only the parameters that require gradients
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] ## get the parameters that have a dimension of 2 or more
    no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2] ## get the parameters that have a dimension of less than 2
    optimizer_groups = [
        {'params': decay_params, 'weight_decay': 0.0},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.Adam(optimizer_groups)#, betas = (0.9, 0.95))
    # optimizer = torch.optim.Adam(model.parameters())

    ## scheduler
    # learning rate
    lr_scheduler_dict = {
        'constant': (LRScheduler, (cfg.lr_scheduler.initial_lr,)),
        'cosine': (CosineLRScheduler, (cfg.lr_scheduler.initial_lr, 
                                       cfg.lr_scheduler.min_lr, 
                                       cfg.lr_scheduler.total_steps, 
                                       cfg.lr_scheduler.warmup_steps))
    }
    func, args = lr_scheduler_dict[cfg.lr_scheduler.name]
    lr_scheduler = func(*args)

    # dropout rate
    dropout_scheduler_dict = {
        'constant': (DropoutScheduler, (cfg.dropout_scheduler.initial_dropout,)),
        'linear': (LinearDropoutScheduler, (cfg.dropout_scheduler.initial_dropout,
                                           cfg.dropout_scheduler.final_dropout,
                                           cfg.dropout_scheduler.total_steps))
    }
    func, args = dropout_scheduler_dict[cfg.dropout_scheduler.name]
    dropout_scheduler = func(*args)


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