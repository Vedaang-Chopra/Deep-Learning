

from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionTransformerConfig:
    '''
    1. hidden_size -> Embedding Dimension (Equivalent) ; Embedding Vector ;
    2. intermediate_size ->  Size of Linear Layer
    
    3. num_hidden_layers -> Number of hidden layers
    4. num_attention_heads -> number of attention heads
    
    5a. num_channels -> Image Channels (RGB) 
    5b. patch_size ->  Patch size (ViT processing)
    
    6. image_size -> Image resolution input (224 * 224 processed)
    7. attention_dropout -> Attention tokens
    8. layer_norm_eps -> 
    9. num_image_tokens -> How many output tokens after ViT
    
    ViT -> Tokens input image, it give a
    '''
    hidden_size: int = 768
    embd_dim:int = 768
    d_model:int = 768
    
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    n_heads: int =12
    
    
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    num_image_tokens: Optional[int] = None