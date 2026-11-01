import torch 
import torch.nn as nn
from torch.nn import functional as F

class Qwen3FFn(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        
        ## Step-1: - Setup all Config needed for this class
        self.emb_dim = config.emb_dim 
        self.hidden_dim = config.hidden_dim
        self.dtype = config.dtype
        
        
        ## Step-2: - Setup all layers 
        
        # 1. 3 Linear Layers 
        self.gate_proj = nn.Linear(self.emb_dim, self.hidden_dim, bias= False, dtype=self.dtype)
        self.up_proj = nn.Linear(self.emb_dim, self.hidden_dim, bias= False, dtype=self.dtype)
        self.down_proj = nn.Linear(self.hidden_dim, self.emb_dim, bias= False, dtype=self.dtype)
                
    
        def forward(self, x: torch.Tensor):
            x_fc1 = self.gate_proj(x)
            x_fc2 = self.up_proj(x)
            x = self.down_proj(F.silu(x_fc1) * x_fc2)
            return x
    
            