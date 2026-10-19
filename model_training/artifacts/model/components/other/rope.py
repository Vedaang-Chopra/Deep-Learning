
import torch
import torch.nn as nn
from torch import Tensor

class RotaryPositionEmbedding(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        # Head Dimension can not be odd as we need 2 group by 2 
        if config.head_dim % 2:
            raise ValueError("Head Dim must be even")
        
        self.head_dim = config.head_dim
        self.context_len = config.context_len
        self.theta = config.theta
        
        # Register precomputed non-persistent cosine and sine tables here.
        ## here we use register_buffer for non trainable values
        freq = self.precompute_theta_pos(self.head_dim, self.context_len, self.theta)
        
        self.register_buffer("cos_emb", freq.cos(), persistent=False)
        self.register_buffer("sin_emb", freq.sin(), persistent=False)
        
    
    def precompute_theta_pos(self, head_dim:int, seq_len:int, theta:float =10000):
        ###
        ### Pre Calculating Theta properly 
        theta_arr = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
        
        ## Calculating all M's 
        pos_arr = torch.arange(0, seq_len)
        
        ## Outer Product for all combinations 
        freqs = torch.einsum('i,j -> ij', pos_arr, theta_arr).float()
        
        return freqs.repeat_interleave(2, dim=-1) 
            
    def forward(self, x: Tensor, position_offset: int = 0) -> Tensor:
            seq_len = x.shape[-2]
            """Apply RoPE to a tensor shaped ``(..., sequence, head_dim)``."""
            
            start= position_offset
            end = position_offset + seq_len
                
            cos_emb_t = self.cos_emb[start:end].unsqueeze(0).unsqueeze(0)
            sin_emb_t = self.sin_emb[start:end].unsqueeze(0).unsqueeze(0)
            x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
            
            x_partner = torch.stack(
                [-x_pairs[..., 1], x_pairs[..., 0]],
                dim=-1
            ).flatten(-2)
            return (x.float() * cos_emb_t + x_partner * sin_emb_t).type_as(x)