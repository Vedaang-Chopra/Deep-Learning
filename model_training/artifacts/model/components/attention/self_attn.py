import torch 
import torch.nn as nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert config.emb_dim % config.n_head ==0 ## THis ensures embedding is split properly across the heads
        
        ## 
        self.c_attn = nn.Linear(config.emb_dim, 3 * config.emb_dim)
        
        # self.c_proj.NANOGPT_SCALE_INIT =1
        
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim)
        ## Regularization 
        self.n_head = config.n_head
        self.emb_dim = config.emb_dim
        
        self.register_buffer("bias", 
                            torch.tril(torch.ones(config.context_len, config.context_len)).view(
                            1, 1, config.context_len, config.context_len
                            ))
        
    def forward(self, x):
        B, T, C = x.size()          # Batch Size, Sequence Length, Embedding Dimensionality
        
        qkv = self.c_attn(x)
        
        q, k, v = qkv.split(self.emb_dim, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) ## (B, nh,  T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) ## (B, nh,  T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) ## (B, nh,  T, hs)
        
        att = q @ k.transpose(-2, -1) 
        att /= math.sqrt(k.size(-1))
        
        att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att@v
        
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
        y = self.c_proj(y)
        
        return y
  


## Fill this properly
# class CausalSelfAttention(nn.Module):
    
    
#     def __init__(self, ):
        
#         pass
    
#     def forward(self, ):
        