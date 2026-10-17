
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        ## Step-1:- Assigning the config parameters
        self.context_length = config.context_length
        self.embd_dim = config.embd_dim
        self.num_heads = config.num_heads
        self.context_length = config.context_length
        
        ## Calculating head_dim
        
        assert self.embd_dim % self.num_heads == 0, "embedding dimension must be divisible by num_heads"
        self.head_dim = self.embd_dim // self.num_heads
        
        d_in , d_out = self.embd_dim, self.embd_dim
        ## Step-3: - Create the Q, K, V Layers
        self.W_query = nn.Linear(d_in, d_out, bias=config.use_attn_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=config.use_attn_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=config.use_attn_bias)
        
        
        
        ## 
        
        
        
        
    def forward(self, x, 
                cache: KVCache | None = None,
                use_cache: bool = False,):
        """Return attention output and, when requested, the appended KV cache.
        
                Cache tensors use shape ``(batch, n_kv_heads, sequence, head_dim)``.
                """
        batch_size, context_len, embd_dim = x.size()
        
        
        
        
        ## Step-1:- Build the Q, K , V matrices 
        Q, K, V = self.q_proj(x)  , self.k_proj(x),  self.v_proj(x)
        
        ## Step-2: - Prepare them for Multi head Attention 
        Q = Q.view(B, S, self.config.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        
        ### Step-2(b) - Preparing KV Cache and Rope
        Q = self.rope(Q, position_offset = past_len)
        K = self.rope(K, position_offset = past_len)
                
        
                
    