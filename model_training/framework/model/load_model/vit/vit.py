from typing import Any

import torch 
import torch.nn as nn 

from torch.nn import functional as F

from model_training.framework.model.components.other.embedding.embedding import PatchEmbedding
from model_training.framework.model.components.other.normalization.layer_norm import LayerNorm
from model_training.framework.model.components.attention.mha import MultiHeadAttention 
from model_training.framework.model.components.ffn.gpt2_ffn import MLP
class VisionTransformer(nn.Module):
    
    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        ### Step-1: - Setup Parameters
        
        self.embd_dim = config.embd_dim
        
        self.embd = PatchEmbedding(config)
        
        self.layer_norm1 = LayerNorm(self.embd_dim)
        self.self_attn = MultiHeadAttention(config)
        self.layer_norm2 = LayerNorm(self.embd_dim)
        self.ffn = MLP(config)
        
        
    def forward(self, x):
        residual = x
        
        x = self.layer_norm1(x)
        attn_out, new_cache = self.self_attn(x)
        x = x + attn_out
        
        residual = x
        x = self.ffn(self.layer_norm2(x))
        
        x = residual + x
        
        return x, new_cache
        
        
        
        
        