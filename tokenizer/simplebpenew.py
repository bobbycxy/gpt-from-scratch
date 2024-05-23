import re
from tokenizer.utils import get_stats, merge

class simplebpenew:
    def __init__(self, cfg):
        # load the text file
        with open(cfg['tokenizer']['vocab_file'], 'r') as f:
            self.text = f.read()

        # check special_tokens_dict
        self.pad_token = b'<pad>'
        self.pad_token_id = 256

        self.unk_token = b'<unk>'
        self.unk_token_id = 257

        self.bos_token = b'<bos>'
        self.bos_token_id = 258

        self.eos_token = b'<eos>'
        self.eos_token_id = 259

        self.mask_token = b'<mask>'
        self.mask_token_id = 260
        
        self.special_tokens_dict = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.mask_token: self.mask_token_id
        }
        self.special_tokens_pattern_re = b'|'.join(re.escape(token) for token in self.special_tokens_dict.keys())

        ## create the properties
        self.vocab_offset = max(self.special_tokens_dict.values()) + 1
        self.vocab_size = cfg['tokenizer']['vocab_size']

        ## create the tokens
        self.tokens = self.text.encode("utf-8") # encode the text as raw bytes. This will be a list of integers in range 0..255
        self.tokens = list(map(int, self.tokens)) # convert to a list of integers for convenience

        ## calculate the number of merges
        self.num_merges = self.vocab_size - self.vocab_offset
        self.ids = list(self.tokens) # copy the tokens so that we don't destroy the original list

        self.merges = {}
        for i in range(self.num_merges):
            stats = get_stats(self.ids)
            top_pair = max(stats, key=stats.get)
            idx = self.vocab_offset + i
            print(f"merge {top_pair} into a new token {idx}")
            self.ids = merge(self.ids, top_pair, idx)
            self.merges[top_pair] = idx

        ## creates the vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)} # create the vocab for the first 256 characters
        self.vocab.update({i: ch for ch, i in self.special_tokens_dict.items()}) # add the special tokens
        for (p0, p1), idx in self.merges.items(): # add the merged tokens
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]



    def pre_tokenize(self, given_text):
        # Convert text to bytes
        text_bytes = given_text.encode('utf-8')
        
        # Regex to split text by special tokens, preserving the tokens in the output
        pattern = b'(' + self.special_tokens_pattern_re + b')'
        # Split text based on the pattern
        tokens = re.split(pattern, text_bytes)
        
        # Remove empty tokens resulting from the split
        tokens = [token for token in tokens if token]
        
        return tokens
    
    
    def encode(self, given_text):
        pre_tokenized_text = self.pre_tokenize(given_text)
        chtoi = {v: k for k, v in self.vocab.items()}
        tokens = []
        for pre_token in pre_tokenized_text:
            if pre_token in self.special_tokens_dict.keys():
                tokens.append([chtoi[pre_token]])
            else:
                tokens.append(list(pre_token))
        
        tokens = sum(tokens, [])

        ## perform the merges; compression
        while len(tokens) >= 2: # iterate until we can't merge anymore
            stats = get_stats(tokens)
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)

        return tokens

    def decode(self, ids):
        '''
        Given a list of ids, this function will return the decoded text.
        '''
        tokens = b"".join(self.vocab[i] for i in ids)
        text = tokens.decode("utf-8", errors = "replace")
        return text