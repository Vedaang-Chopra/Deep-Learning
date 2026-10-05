import torch
import torch.nn as nn
from torch.nn import functional as F



from old_model_training.model_training.artifacts.model.components.attention.self_attn import CausalSelfAttention
from old_model_training.model_training.artifacts.model.components.ffn.gpt2_ffn import MLP



class Block(nn.Module):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP (config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x    
    