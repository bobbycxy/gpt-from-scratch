import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import hydra
from hydra import initialize, compose

import torch
import torch.multiprocessing as mp
from trainers.build_trainer import Trainer, ddp_setup
from torch.distributed import destroy_process_group

def ddp_main(rank, world_size, cfg):
    ddp_setup(rank=rank, world_size=world_size)
    trainer = Trainer(cfg, rank)
    trainer.train()
    destroy_process_group()

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_main, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
