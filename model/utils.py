import torch
import inspect


def build_optimizer(model, weight_decay, betas):
    '''
    Build the optimizer for the model
    '''
    param_dict = {pn: p for pn, p in model.named_parameters()} ## get each layer of the model
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} ## get only the parameters that require gradients
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] ## get the parameters that have a dimension of 2 or more
    no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2] ## get the parameters that have a dimension of less than 2
    optimizer_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    ## use fused adam if available
    fused_available = 'fused' in inspect.signature(torch.optim.Adam).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.Adam(optimizer_groups, betas = betas, **extra_args)

    return optimizer