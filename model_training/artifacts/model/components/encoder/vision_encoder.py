import torch
import torch.nn as nn

from model_training.artifacts.model.components.encoder.ve_config import SiglipVisionConfig


class SigLipVisionModel(nn.Module):
    
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)
        
    
    
    def forward(self, pixel_values) -> Tuple:
        
        
        # (B, Channels, Height, Width) -> (B, Num_patches, Embd_dim)
        return self.vision_model(pixel_values)