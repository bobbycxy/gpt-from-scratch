'''
Module for simple byte pair encoding tokenizer
'''
from tokenizer.utils import get_stats, merge

class character:
    def __init__(self, cfg):
        with open(cfg['tokenizer']['vocab_file'], 'r') as f:
            self.text = f.read()
        self.char = sorted(list(set(self.text)))
        self.vocab_size = len(self.char)

    def decode(self, ids):
        '''
        Given a list of ids, this function will return the decoded text.
        '''
        idx_to_char = {i:ch for i,ch in enumerate(self.char)}
        text = ''.join([idx_to_char[i] for i in ids])
        return text        


    def encode(self, text):
        '''
        Given a text, this function will return the encoded ids.
        '''
        char_to_idx = {ch:i for i,ch in enumerate(self.char)}
        out = [char_to_idx[ch] for ch in text]
        return out

# Path: tokenizer/utils.py