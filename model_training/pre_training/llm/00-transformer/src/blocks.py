import torch
import torch.nn as nn
from layer_norm import LayerNorm
from attention import MultiHeadAttention

class FeedForward(nn.Module):
    
    '''
    This is the feedforward block. 2 linear layers 
    Linear layer, Relu,  Linear Layer and then Relu (all with biases)
    d_model -> Is it embedding dimension, -> Paper said 512
    d_ff -> Linear layer nodes -> 512 (No of nodes)
    
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))




class ResidualConnection(nn.Module):
    '''
    Creating this block for residual connections
    '''
    
    def __init__(self, dropout):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        

class EncoderBlock(nn.Module):
    def __init__(self, 
                attention_block:MultiHeadAttention,
                feed_forward_block:FeedForward,
                dropout: float, 
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.attn_block = attention_block
        self.ffn_block = feed_forward_block
        
        self.resd_conn = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(2)
            ])

    def forward(self, x, src_mask):
        x = self.resd_conn[0](x, lambda x: self.attn_block(x, x, x, src_mask))

        x = self.resd_conn[1](x, self.ffn_block)
        return x



class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        
        self.layers = layers
        self.norm  = LayerNorm()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)
        
        
    