'''
Utility functions that are used in the trainers objects
'''

import os
import torch
import time

def save_checkpoint(model, optimizer, epoch, loss, path):
    '''
    Save the checkpoint to the path. This function will save the model and optimizer state_dict, epoch and loss.
    '''
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} in {path}")

def load_checkpoint(model, optimizer, path):
    '''
    Load the checkpoint from the path. This function will load the model and optimizer state_dict and return the epoch.
    '''
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}, with eval loss of {loss}")
    return epoch

def ensure_dir(path):
    '''
    Ensure that the directory exists, if not create it
    '''
    os.makedirs(path, exist_ok=True)
    print(f"Directory {path} exists or created.")

def generate_run_name(languagemodel_name, model_name, dataset_name, tokenizer_name, learning_rate, batch_size, max_iters):
    '''
    Generate a name for the wandb run
    '''
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = f"{languagemodel_name}_{model_name}_{dataset_name}_{tokenizer_name}_lr{learning_rate}_bs{batch_size}_iters{max_iters}_{timestamp}"
    return run_name

def init_wandb(cfg, model):
    '''
    Initialize wandb
    '''
    import wandb
    from omegaconf import OmegaConf
    run_name = generate_run_name(cfg.languagemodel.name, cfg.model.name, cfg.train_file, cfg.tokenizer.name, cfg.learning_rate, cfg.batch_size, cfg.max_iters)
    wandb.init(project=cfg.wandb_project, name=run_name, config = OmegaConf.to_container(cfg))
    wandb.watch(model)

def log_wandb(losses):
    '''
    Log the losses to wandb
    '''
    import wandb
    wandb.log({'train_loss': losses['train'], 'val_loss': losses['val']})
