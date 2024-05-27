'''
Create a data loader for the model
'''

import torch
from tokenizer import *
from trainers.utils import masked_lm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_dict = {
    'character': character,
    'simplebpe': simplebpe,
    'characternew': characternew,
    'simplebpenew': simplebpenew,
    'bpe': bpe
}

class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        with open(cfg['train_file'],'r') as f:
            self.text = f.read()
        self.block_size = cfg['block_size']
        self.batch_size = cfg['batch_size']

        self.tokenizer = tokenizer_dict.get(cfg['tokenizer']['name'])(cfg)
        self.data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)

        train_val = int(0.9*len(self.data)) ## 90% train and val, 10% test
        n = int(cfg['train_size']*train_val)
        self.train_data, self.val_data, self.test_data = self.data[:n], self.data[n:train_val], self.data[train_val:]

    def load(self, split):
        data_dict = {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data
        }
        data = data_dict[split]
        idx = torch.randint(0, len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i     :   i+self.block_size   ] for i in idx])

        if split == 'test':
            return x.to(device), None
        
        ## apply mlm if masked language model is true
        if self.cfg['languagemodel']['masked_lm']: 
            x, y = masked_lm(x, self.tokenizer, self.cfg['languagemodel']['mlm_prob'])
        else: ## else, shift the data by one
            y = torch.stack([data[i+1   :   i+self.block_size+1 ] for i in idx])
        
        return x.to(device), y.to(device)
        