import torch
from evals.inference import *
from trainers.dataloader import DataLoader

import hydra

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    ## load model
    tokenizer, model, optimizer = load_model(cfg)
    
    ## init the data loader, and get the test_data
    data_loader = DataLoader(cfg = cfg)
    test_data, _ = data_loader.load('test')

    ## inference
    inference_dict = {
        'masked': predict_mask,
        'causal': generate_text
    }
    ## get a random sample from the test_data
    random_idx = torch.randint(0, len(test_data), (1,)).item()
    test_sample = test_data[random_idx]

    ## run inference
    inference_dict[cfg.languagemodel.name](tokenizer, model, test_sample)

if __name__ == "__main__":
    main()