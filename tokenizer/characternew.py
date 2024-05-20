import re

class characternew:
    def __init__(self, cfg):
        ## Load the text file
        with open(cfg['tokenizer']['vocab_file'], 'r') as f:
            self.text = f.read()
        
        ## check special_tokens_dict
        self.pad_token = '<pad>'
        self.pad_token_id = 0

        self.unk_token = '<unk>'
        self.unk_token_id = 1

        self.bos_token = '<bos>'
        self.bos_token_id = 2

        self.eos_token = '<eos>'
        self.eos_token_id = 3

        self.mask_token = '<mask>'
        self.mask_token_id = 4
        
        self.special_tokens_dict = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.mask_token: self.mask_token_id
        }

        ## Create the character to index and index to character mapping
        self.chars = sorted(list(set(self.text)))
        self.vocab_offset = max(self.special_tokens_dict.values()) + 1
        self.vocab = {char: index + self.vocab_offset for index, char in enumerate(self.chars)}
        self.vocab.update(self.special_tokens_dict)
        self.vocab_size = len(self.vocab)

        ## Create the character to index and index to character mapping
        self.char_to_index = {char: index for char, index in self.vocab.items()}
        self.index_to_char = {index: char for char, index in self.vocab.items()}

        ## Add special tokens to the mapping

    def pre_tokenize(self, given_text):
        '''
        This function will split the text into tokens while keeping the special tokens intact.
        '''

        ## escape special tokens for regex
        escaped_tokens = [re.escape(token) for token in self.special_tokens_dict.keys()]
        self.special_tokens_pattern = re.compile('|'.join(escaped_tokens))
        self.word_space_pattern = re.compile(r'\s*\S+|\s+')

        # Split the text using the precompiled special tokens pattern
        parts = self.special_tokens_pattern.split(given_text)
        matches = self.special_tokens_pattern.findall(given_text)

        # Recombine parts and matches
        tokens = []
        for i, part in enumerate(parts):
            if part:
                # Split non-special parts further into words and spaces
                tokens.extend(self.word_space_pattern.findall(part))
            if i < len(matches):
                # Append the special token
                tokens.append(matches[i])

        return tokens
    
    def encode(self, given_text):
        '''
        Encodes the given text into a list of indices
        '''
        pre_tokenized_text = self.pre_tokenize(given_text)
        splits =[[word] if any(substring in word for substring in self.special_tokens_dict.keys()) else [l for l in word] for word in pre_tokenized_text]
        flattened_splits = sum(splits,[])
        res = [self.char_to_index.get(char, self.char_to_index[self.unk_token]) for char in flattened_splits]
        return res

    def decode(self, indices):
        '''
        Decodes the given indices into text
        '''

        return ''.join([self.index_to_char.get(index, self.unk_token) for index in indices])
