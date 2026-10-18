import torch 
import torch.nn as nn



class SimpleEmbedding(nn.module):
    
    def __init__(self, config):
        
        ### Step-1: - Assign Variables
        self.vocab_size = config.vocab_size
        self.embd_dim = config.embd_dim 
        
        # Step-2: - Declare an Embedding Layer
        self.embd_layer = nn.Embedding(self.vocab_size, self.embd_dim)
        
        
        
        def forward(self, x_in:torch.Tensor):            
            embed_val = self.embd_layer(x_in)
            
            return self.embd_layer



class TokenandPosEmbedding(nn.module):
    
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
            