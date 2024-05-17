'''
Module for simple byte pair encoding tokenizer
'''
from tokenizer.utils import get_stats, merge

class simplebpe:
    def __init__(self, cfg):
        with open(cfg['tokenizer']['vocab_file'], 'r') as f:
            self.text = f.read()

        self.tokens = self.text.encode("utf-8") # encode the text as raw bytes. This will be a list of integers in range 0..255
        self.tokens = list(map(int, self.tokens)) # convert to a list of integers for convenience

        self.vocab_size = cfg['tokenizer']['vocab_size']
        self.num_merges = self.vocab_size - 256 # 256 is the number of single byte tokens
        self.ids = list(self.tokens) # copy the tokens so that we don't destroy the original list

        self.merges = {} # store the merges
        for i in range(self.num_merges): # iterate over the number of merges we want to do
            stats = get_stats(self.ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merge {top_pair} into a new token {idx}")
            self.ids = merge(self.ids, top_pair, idx)
            self.merges[top_pair] = idx

        ## creates the vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)} # store the vocab for the first 256 tokens
        for (p0, p1), idx in self.merges.items(): # iterate over the merges
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        ## save the compression rate
        self.compression_rate = f"Provided text's compression rate with {self.num_merges} new tokens: {len(self.tokens) / len(self.ids):.2f}x"



    def decode(self, ids):
        '''
        Given a list of ids, this function will return the decoded text.
        '''
        tokens = b"".join(self.vocab[i] for i in ids)
        text = tokens.decode("utf-8", errors = "replace")
        return text
    


    def encode(self, text):
        '''
        Given a text, this function will return the encoded ids.
        '''
        tokens = list(text.encode("utf-8")) # encode the text as raw bytes
        while len(tokens) >= 2: # iterate until we can't merge anymore
            stats = get_stats(tokens)
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        
        return tokens



# Path: tokenizer/utils.py