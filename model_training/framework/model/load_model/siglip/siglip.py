import torch
import torch.nn as nn
from typing import Tuple


from model_training.framework.model.load_model.siglip.config import SiglipVisionConfig
from model_training.framework.model.components.other.normalization.layer_norm import LayerNorm

from model_training.framework.model.components.attention.mha import MultiHeadAttention as SiglipAttention
from model_training.framework.model.components.ffn.siglip_ffn import SiglipMLP
from model_training.framework.model.components.other.embedding.embedding import SiglipEmbedding




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
                
        




class SigLipVisionModel(nn.Module):
    
    def __init__(self, config:SiglipVisionConfig):
            super().__init__()
            
            self.config = config
            self.embd_dim = config.hidden_size
            
            self.embeddings = SiglipEmbedding(config)
            self.encoder = SiglipEncoder(config)
            self.post_layernorm = nn.LayerNorm(self.embd_dim, eps = config.layer_norm_eps)
            
            
    def forward(self, pixel_values:torch.Tensor) -> torch.Tensor:
        # Input - [Batch_Size, Channels, Height, Width] -> 
        # Output - [Batch_Size, Num_Patches, Embed_Dim]
                
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
         # (B, Channels, Height, Width) -> (B, Num_patches, Embd_dim)
        return last_hidden_state
    
        
