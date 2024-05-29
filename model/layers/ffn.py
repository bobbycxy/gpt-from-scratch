'''
Feedforward network layer for the transformer model
multilayer perceptron with ReLU activation function
'''

import torch.nn as nn
from torch.functional import F

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed), # projection layer going back into the residual pathway
            nn.Dropout()
        )
    
    def forward(self, x):
        return self.net(x)