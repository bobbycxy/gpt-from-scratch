'''
Helper functions for the trainers objects
'''

import torch

from trainers.scheduler import *

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


def masked_lm(input_ids, tokenizer, mlm_prob):
    '''
    Masked language model
    '''
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id
    special_tokens_dict = tokenizer.special_tokens_dict

    mask = torch.rand(input_ids.size()) < mlm_prob # create a mask of true and false
    mask &= input_ids != pad_token_id # do not mask padding
    mask &= input_ids != mask_token_id # do not mask mask token
    mask &= input_ids != unk_token_id # do not mask unk token
    ## typically, there won't be any bos or eos tokens in the input_ids. Most likely might have padding tokens or mask tokens

    ## create clones
    mlm_data = input_ids.clone()
    labels = input_ids.clone()

    ## get the indices of the mask
    mask_idx = mask.nonzero(as_tuple=True)

    ## randomly mask tokens
    mask_idx_shuffle = torch.randperm(mask_idx[0].shape[0])

    ## get the tokens for each type of masking (mask, random, keep)
    tomask_idx = mask_idx_shuffle[:int(mask_idx[0].shape[0] * 0.8)]
    torandom_idx = mask_idx_shuffle[int(mask_idx[0].shape[0] * 0.9):]

    ## mask the tokens
    mlm_data[mask_idx[0][tomask_idx], mask_idx[1][tomask_idx]] = mask_token_id
    mlm_data[mask_idx[0][torandom_idx], mask_idx[1][torandom_idx]] = generate_random_token_ids((torandom_idx.shape[0],), vocab_size, special_tokens_dict) 

    ## create the labels
    labels[~mask] = -100

    return mlm_data, labels 


def generate_random_token_ids(shape, vocab_size, special_tokens_dict):
    '''
    Takes a shape and randomly generates token ids within the vocab size
    but not including pad_token_id or mask_token_id.
    '''

    random_token_ids = torch.randint(0, vocab_size, shape)
    # Ensure random_token_ids do not contain special token ids, e.g. pad_token_id, mask_token_id, etc.
    invalid_ids = special_tokens_dict.values()
    for i in range(random_token_ids.numel()):
        while random_token_ids.view(-1)[i].item() in invalid_ids: ## iteratively check if the token is in the invalid_ids
            random_token_ids.view(-1)[i] = torch.randint(0, vocab_size, (1,))
    return random_token_ids


def build_lr_scheduler(cfg):
    ## using lambda so that the function is not called until it is needed
    lr_scheduler_dict = {
        'constant': lambda cfg: LRScheduler(cfg.scheduler.learningrate.initial_lr),
        'cosine': lambda cfg: CosineLRScheduler(
            cfg.scheduler.learningrate.initial_lr,
            cfg.scheduler.learningrate.min_lr,
            cfg.scheduler.learningrate.total_steps,
            cfg.scheduler.learningrate.warmup_steps
        )
    }

    try:
        lr_scheduler = lr_scheduler_dict[cfg.scheduler.learningrate.name](cfg)
    except KeyError:
        raise ValueError(f"Invalid LR scheduler name: {cfg.scheduler.learningrate.name}")
    
    return lr_scheduler

def build_dropout_scheduler(cfg):
    ## using lambda so that the function is not called until it is needed
    dropout_scheduler_dict = {
        'constant': lambda cfg: DropoutScheduler(cfg.scheduler.dropout.initial_dropout),
        'linear': lambda cfg: LinearDropoutScheduler(
            cfg.scheduler.dropout.initial_dropout,
            cfg.scheduler.dropout.final_dropout,
            cfg.scheduler.dropout.total_steps
        )
    }

    try:
        dropout_scheduler = dropout_scheduler_dict[cfg.scheduler.dropout.name](cfg)
    except KeyError:
        raise ValueError(f"Invalid Dropout scheduler name: {cfg.scheduler.dropout.name}")
    
    return dropout_scheduler