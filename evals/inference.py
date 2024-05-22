'''
A script for running inference on a trained model.
'''

import torch
from torch.functional import F
from tokenizer import character, simplebpe, characternew
from model import GPT2, simplebigram
from utils import load_checkpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(cfg):    
    '''
    Takes the config parameters (cfg) and loads the tokenizer, model, and optimizer.
    '''

    ## load tokenizer
    tokenizer_dict = {
        'character': character,
        'simplebpe': simplebpe,
        'characternew': characternew
    }
    tokenizer = tokenizer_dict.get(cfg['tokenizer']['name'])(cfg)
    cfg['vocab_size'] = tokenizer.vocab_size

    ## load model
    model_dict = {
        'simplebigram': simplebigram,
        'gpt2': GPT2
    }
    model = model_dict[cfg.model.name](cfg = cfg).to(device)

    ## init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    ## load checkpoint
    load_checkpoint(model, optimizer, cfg.languagemodel.model_ckpt)

    return tokenizer, model, optimizer


def predict_mask(tokenizer, model, input_ids):
    '''
    Takes a single input indices, encodes it, randomly masks 1 token, and gets the model 
    to make a prediction. This is used in tandem with the masked language model.
    '''
    ## reshape the input_ids
    input_ids = input_ids.unsqueeze(0) # (1, seq_len)

    ## encode text
    og_sentence_id = input_ids.to(device)
    
    ## mask a token
    idx = torch.randint(0, og_sentence_id.size(1), (1,)).item()
    ms_token_id = og_sentence_id[0, idx].item()
    ms_token = tokenizer.decode([ms_token_id])

    ## mask the input_ids
    ms_sentence_id = og_sentence_id.clone()
    ms_sentence_id[0, idx] = tokenizer.mask_token_id

    ## create clone for pred_sentence_id
    pr_sentence_id = ms_sentence_id.clone()

    ## predict the masked token
    with torch.no_grad():
        model.eval() # set model to eval mode
        logits, _ = model(ms_sentence_id)
        model.train() # set model back to train mode

        probs = F.softmax(logits, dim=-1) # (1, seq_len, vocab_size)
        pr_token_id = torch.argmax(probs[0, idx]).item() # get the token with the highest probability

    ## decode the predicted token id
    predicted_token = tokenizer.decode([pr_token_id])
    pr_sentence_id[0, idx] = pr_token_id

    ## print results
    print('RESULT:', ms_token_id == pr_token_id)
    print('----------')
    print('ACTUAL:', tokenizer.decode(og_sentence_id[0].tolist()))
    print('----------')
    print('MASKED:', tokenizer.decode(ms_sentence_id[0].tolist()))
    print('----------')
    print('PREDICTED:', tokenizer.decode(pr_sentence_id[0].tolist()))
    print('----------')
    print(f"Original: {ms_token}, Predicted: {predicted_token}")

    return int(ms_token_id == pr_token_id)


def generate_text(tokenizer, model, input_ids, max_new_tokens=250):
    '''
    Takes a context and generates new text. This is used in tandem
    with the causal language model.
    '''
    ## reshape the input_ids
    input_ids = input_ids.unsqueeze(0) # (1, seq_len)

    ## decode text
    context = tokenizer.decode(input_ids[0].tolist())

    ## generate new text
    out = model.generate(input_ids, max_new_tokens=max_new_tokens)

    ## decode the generated text
    res = tokenizer.decode(out[0].tolist())

    print('CONTEXT:', context)
    print('----------')
    print('GENERATED:', res)

    
