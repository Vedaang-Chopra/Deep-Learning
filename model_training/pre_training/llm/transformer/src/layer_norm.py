import torch 
import torch.nn as nn 



class LayerNorm(nn.Module):
    '''
    
    We introduce GAMMA(multiplicative) and BETA(additive) in normalization to amplify the values, so that everything does not remain in 0 and 1. These will cause fluctuations when learning in data. These 2 are learnable parameters
    
    '''
    
    def __init__(self, eps = 10**-6):
        super().__init__()
        self.eps = eps ## Avoid dividing zero
        self.gamma = nn.Parameter(torch.ones(1))     ## Multiplicative
        self.beta = nn.Parameter(torch.ones(1))     ## Additive
        
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_hat = (x - mean ) / (std + self.eps)
        layer_norm = self.gamma * x_hat + self.beta
        return layer_norm
        


