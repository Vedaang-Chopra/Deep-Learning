import torch 
import torch.nn as nn

from typing import Tuple

from model_training.framework.model.components.block.encoder.ve_config import SiglipVisionConfig



class SimpleEmbedding(nn.Module):
    
    def __init__(self, config):
        
        ### Step-1: - Assign Variables
        self.vocab_size = config.vocab_size
        self.embd_dim = config.embd_dim 
        
        # Step-2: - Declare an Embedding Layer
        self.embd_layer = nn.Embedding(self.vocab_size, self.embd_dim)
        
        
        
        def forward(self, x_in:torch.Tensor):            
            embed_val = self.embd_layer(x_in)
            
            return self.embd_layer



class TokenandPosEmbedding(nn.Module):
    
    def __init__(self, config):
        
        ### Step-1: - Assign Variables
        self.vocab_size = config.vocab_size
        self.embd_dim = config.embd_dim 
        
        # Step-2: - Declare an Embedding Layer
        self.token_layer = nn.Embedding(self.vocab_size, self.embd_dim)
        self.position_layer = nn.Embedding(self.vocab_size, self.embd_dim)
        
        
        
        def forward(self, x_in:torch.Tensor):            
            embed_val = self.token_layer(x_in)
            
            return self.embd_layer




class SiglipEmbedding(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        
        
        ## Step-1: - Isolate the configs to use
        self.embd_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embd_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = "valid"
        )
        
        self.num_patches = (self.image_size // self.patch_size) **2
        self.num_positions = self.num_patches
        
        self.position_embedding = nn.Embedding(
            self.num_positions , self.embd_dim
        )
        self.register_buffer("position_ids", 
                            torch.arange(self.num_positions).expand(1, -1), 
                            persistent = False)
        
        
    
    def forward(self, pixel_values:torch.FloatTensor ) -> torch.Tensor:
        
        batch_size, channels, height, width = pixel_values.size()
        
        embeddings = self.patch_embedding(pixel_values)
        print(f"Step-1: {embeddings.shape}")
        
        embeddings = embeddings.flatten(2)
        print(f"Step-2: {embeddings.shape}")
        
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        # This transpose ensures we follow conventional attention format
        embeddings = embeddings.transpose(1, 2)
        
        
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
        
        