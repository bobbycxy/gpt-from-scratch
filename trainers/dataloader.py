'''
Create a data loader for the model
'''

import torch
from tokenizer import *
from trainers.utils import masked_lm
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_dict = {
    'character': character,
    'simplebpe': simplebpe,
    'characternew': characternew,
    'simplebpenew': simplebpenew,
    'bpe': bpe
}

class DataLoader:
    '''
    Andrej Karpathy's DataLoader
    '''
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
    
class CustomDataset(Dataset):
    '''
    Custom dataset for the model that will be used in the DataLoader.
    This custom dataset will need to have the following methods:
    __len__ : returns the length of the dataset
    __getitem__ : returns the data given an index

    '''

    def __init__(self, cfg, split):
        self.cfg = cfg
        with open(cfg['train_file'], 'r') as f:
            self.text = f.read()
        self.block_size = cfg['block_size']
        self.batch_size = cfg['batch_size']

        self.tokenizer = tokenizer_dict.get(cfg['tokenizer']['name'])(cfg)
        print(f'...Encoding {split}')
        self.data = torch.tensor(self.tokenizer.encode(self.text), dtype=torch.long)
        

        train_val = int(0.9 * len(self.data))  # 90% train and val, 10% test
        n = int(cfg['train_size'] * train_val)
        
        data_splits = [
            lambda: self.data[:n],  # train
            lambda: self.data[n:train_val],  # val
            lambda: self.data[train_val:]  # test
        ]

        self.data = data_splits[{'train': 0, 'val': 1, 'test': 2}[split]]()
        self.split = split

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        
        if self.split == 'test':
            return x.to(device), None

        if self.cfg['languagemodel']['masked_lm']:
            x, y = masked_lm(x, self.tokenizer, self.cfg['languagemodel']['mlm_prob'])
        else:
            y = self.data[idx + 1:idx + self.block_size + 1]

        return x.to(device), y.to(device)
        