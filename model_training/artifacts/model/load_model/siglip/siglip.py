import torch
import torch.nn as nn

from typing import Tuple

from model_training.artifacts.model.components.block.encoder.ve_config import SiglipVisionConfig



from model_training.artifacts.model.components.other.embedding.embedding import SiglipEmbedding


from model_training.artifacts.model.components.block.encoder.siglip_block import SiglipEncoder



class SigLipVisionTransformer(nn.Module):
    
    def __init__(self, config:SiglipVisionConfig):
            super().__init__()
            
            self.config = config
            self.embd_dim = config.hidden_size
            
            self.embeddings = SiglipEmbedding(config)
            self.encoder = SiglipEncoder(config)
            
            
            self.post_layernorm = nn.Layernorm(self.embd_dim, eps = config.layer_norm_eps)
            
            
    def forward(self, pixel_values:torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.encoder(input_embeds = hidden_states)
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
        
    
class SigLipVisionModel(nn.Module):
    
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)
        
    def forward(self, pixel_values) -> Tuple:

        # Input - [Batch_Size, Channels, Height, Width] -> 
        # Output - [Batch_Size, Num_Patches, Embed_Dim]
        
        # (B, Channels, Height, Width) -> (B, Num_patches, Embd_dim)
        return self.vision_model(pixel_values)



        
