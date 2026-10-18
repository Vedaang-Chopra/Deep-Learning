class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()
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
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

