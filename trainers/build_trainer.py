import os

import torch.utils
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
import torch.nn as nn
import torch.functional as F

import hydra

from trainers.dataloader import CustomDataset
from model import GPT2, SimpleBigram, build_optimizer

from utils import save_checkpoint, load_checkpoint, ensure_dir
from trainers.utils import build_dropout_scheduler, build_lr_scheduler
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from hydra import initialize, compose

## new
from itertools import islice

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import torch.distributed as dist


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(self, cfg, gpu_id):
        self.cfg = cfg
        self.gpu_id = gpu_id ## unique identifier of each process

        ## dataloader
        self.train_dataset = CustomDataset(cfg = cfg, split = 'train')
        self.val_dataset = CustomDataset(cfg = cfg, split = 'val')
        cfg['vocab_size'] = self.train_dataset.tokenizer.vocab_size
        
        ## model
        self.model_dict = {
            'simplebigram': SimpleBigram,
            'gpt2': GPT2
        }
        self.model = self.model_dict[cfg.model.name](cfg = cfg).to(device)
        self.model = DDP(self.model, device_ids=[gpu_id], find_unused_parameters=True) ## DDP
        
        ## optimizer
        self.optimizer = build_optimizer(self.model, 0.0, (0.9,0.999))
        
        ## schedulers
        self.lr_scheduler = build_lr_scheduler(cfg)
        self.dropout_scheduler = build_dropout_scheduler(cfg)
        
        ## ensure checkpoint directory
        ensure_dir(cfg.model_ckpt_dir)
        self.best_val_loss = float('inf')
        self.checkpoint_path = f'{cfg.model_ckpt_dir}/{cfg.model_ckpt}'

    def _prepare_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False, # was True if no sampler
            num_workers=0,
            sampler=DistributedSampler(dataset) ## DDP
        ) 
    
    def _run_batch(self, xb, yb, train: bool = True):
        ## torch.amp.autocast is used for mixed precision training (fp16) 
        ## torch.set_grad_enabled is used to enable or disable gradient computation
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = self.model(xb, yb)

        ## zero the gradients of the optimizer before running the backward pass
        ## only run the backward pass and update the weights if training
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return loss.item()
    
    @torch.no_grad()
    def _estimate_loss(self, model, train_loader, val_loader, eval_iters):
        out = {}
        model.eval()

        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = torch.zeros(eval_iters)

            ## measure loss up to eval_iters
            for i, (xb, yb) in enumerate(islice(loader, eval_iters)):
                losses[i] = self._run_batch(xb, yb, train=False)
            avg_loss = losses.mean().item()

            ## gather all losses
            all_losses = torch.tensor([avg_loss], device=device)
            dist.all_reduce(all_losses, op=dist.ReduceOp.SUM)

            ## average the losses
            out[split] = all_losses.item() / dist.get_world_size()

        model.train()
        return out
    
    def train(self):

        ## wandb setup
        if self.cfg.wandb_log and self.gpu_id == 0:
            from utils import init_wandb
            init_wandb(self.cfg, self.model)

        ## prepare dataloaders
        train_loader = self._prepare_dataloader(self.train_dataset)
        val_loader = self._prepare_dataloader(self.val_dataset)

        ## training loop
        iter_num = 0
        while iter_num < self.cfg.max_iters:
            for xb, yb in train_loader:
                
                ## update learning rate and dropout rate
                self.lr_scheduler.step(self.optimizer)
                self.dropout_scheduler.step(self.model)

                ## estimate loss and log the loss every eval_interval
                if iter_num % self.cfg.eval_interval == 0:
                    out = self._estimate_loss(self.model, train_loader, val_loader, self.cfg.eval_iters)
                    if dist.get_rank() == 0:
                        print(f'Iter {iter_num}, Train loss: {out["train"]}, Val loss: {out["val"]}')

                    ## wandb logging                  
                    if self.cfg.wandb_log and self.gpu_id == 0:
                        from utils import log_wandb
                        log_wandb({
                            'epoch': iter_num,
                            'train_loss': out["train"],
                            'val_loss': out["val"],
                            'lr': self.optimizer.param_groups[0]['lr'],
                            'dropout_rate': np.mean([module.p for module in self.model.modules() if isinstance(module, nn.Dropout)])
                        })

                    ## Save the model checkpoint if the validation loss is the best we have seen so far
                    if out['val'] < self.best_val_loss and self.gpu_id == 0:
                        self.best_val_loss = out['val']
                        save_checkpoint(self.model, self.optimizer, iter_num, self.best_val_loss, self.checkpoint_path)

                ## run the batch and update the weights every iteration
                self._run_batch(xb, yb, train=True)

                ## increment the iteration number
                iter_num += 1

                ## break if max_iters is reached
                if iter_num >= self.cfg.max_iters:
                    break