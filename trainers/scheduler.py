'''
Schedulers for learning rate and dropout

Learning rate schedulers are used to adjust the learning rate during training.
We typically start with a high learning rate and then decrease it as the training progresses.

Dropout schedulers are used to adjust the dropout rate during training.
We typically start with a low dropout rate and then increase it as the training progresses.
'''
import torch.nn as nn
import math

class LRScheduler:
    '''constant learning rate scheduler'''
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr
        self.current_step = 0
    
    def get_lr(self):
        return self.initial_lr
    
    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def step(self, optimizer):
        self.current_step += 1
        lr = self.get_lr()
        self.set_lr(optimizer, lr)

class CosineLRScheduler(LRScheduler):
    '''cosine learning rate scheduler'''
    def __init__(self, initial_lr, min_lr, total_steps, warmup_steps = 0):
        super().__init__(initial_lr)
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / self.warmup_steps * self.initial_lr
        elif self.current_step < self.total_steps:
            # Cosine learning rate decay
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        else:
            return self.min_lr
        
class DropoutScheduler:
    '''constant dropout rate scheduler'''
    def __init__(self, initial_dropout):
        self.initial_dropout = initial_dropout
        self.current_step = 0

    def get_dropout(self):
        return self.initial_dropout
    
    def set_dropout(self, model, dropout):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    def step(self, model):
        self.current_step += 1
        dropout = self.get_dropout()
        self.set_dropout(model, dropout)

class LinearDropoutScheduler(DropoutScheduler):
    '''linear dropout rate scheduler'''
    def __init__(self, initial_dropout, final_dropout, total_steps):
        super().__init__(initial_dropout)
        self.final_dropout = final_dropout
        self.total_steps = total_steps

    def get_dropout(self):
        if self.current_step < self.total_steps:
            return self.initial_dropout + (self.final_dropout - self.initial_dropout) * self.current_step / self.total_steps
        else:
            return self.final_dropout

# Path: utils.py