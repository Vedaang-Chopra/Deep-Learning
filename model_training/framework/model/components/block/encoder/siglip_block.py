

import torch
import torch.nn as nn



from typing import Tuple
from model_training.framework.model.components.block.encoder.ve_config import SiglipVisionConfig


from model_training.framework.model.components.attention.mha import MultiHeadAttention as SiglipAttention


from model_training.framework.model.components.ffn.siglip_ffn import SiglipMLP

from model_training.framework.model.components.attention.config import TransformerConfig

'''Embedding -> Attention -> FFn -> Layer norm'''
class SiglipEncoderLayer(nn.Module):

     
        def __init__(self, config:SiglipVisionConfig):
                super().__init__()
                
                self.config = config
                
                ## Extract all relevant Configs
                self.embd_dim = config.hidden_size
                self.embd_dim = config.hidden_size
                
                transformer_config = TransformerConfig(
                max_seq_len=(config.image_size // config.patch_size) ** 2,
                d_model=config.hidden_size,
                n_layers=config.num_hidden_layers,
                n_heads=config.num_attention_heads,
                n_kv_heads=config.num_attention_heads,
                d_ff=config.intermediate_size,
                rms_eps=config.layer_norm_eps,
                dropout=config.dropout,
                use_rope=True,
                )
                self.self_attn = SiglipAttention(transformer_config)
                
                self.layer_norm1 = nn.LayerNorm(self.embd_dim, eps= config.layer_norm_eps)
                
                self.mlp = SiglipMLP(config)
                self.layer_norm2 = nn.LayerNorm(self.embd_dim, eps= config.layer_norm_eps)
            
            
         
        def forward(self, hidden_states:torch.Tensor) -> torch.Tensor:
                
                residual = hidden_states
                
                
                hidden_states = self.layer_norm1(hidden_states)
                hidden_states, _ = self.self_attn(hidden_states)
                
                hidden_states= residual + hidden_states
                
                
                residual = hidden_states
                
                hidden_states = self.layer_norm2(hidden_states)
                
                hidden_states = self.mlp(hidden_states)
                
                hidden_states = residual + hidden_states
                
                
                return hidden_states



class SiglipEncoder(nn.Module):
        
        def __init__(self, config:SiglipVisionConfig):
            super().__init__()
            
            self.config = config
            self.layers = nn.ModuleList(
                    [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers) ]
            )
            
        def forward(self, input_embeds:torch.Tensor) -> torch.Tensor:
                hidden_states = input_embeds

                for encoder_layer in self.layers:
                        hidden_states = encoder_layer(hidden_states)
                return hidden_states
                
        