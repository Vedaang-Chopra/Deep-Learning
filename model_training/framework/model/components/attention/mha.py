
import torch
import torch.nn as nn
from typing import TypeAlias
from torch.nn import functional as F


from model_training.framework.model.components.other.rope import RotaryPositionEmbedding
from model_training.framework.model.components.attention.config import TransformerConfig

KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        
        assert config.n_heads % config.n_kv_heads == 0, "Number of query heads must be divisible by KV heads"
        
        
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        
        if self.config.use_rope:
            self.rope = RotaryPositionEmbedding(config.rope_config)
        
        self.dropout = nn.Dropout(config.dropout)
        
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.max_seq_len, 
                    config.max_seq_len).view(1, 1, config.max_seq_len, config.max_seq_len)
                    ), persistent = False)
        self.n_rep = self.config.n_heads // self.config.n_kv_heads
        self.attn = None
        self.o_proj.NANOGPT_SCALE_INIT = 1
        
        
        
    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """Return attention output and, when requested, the appended KV cache.

        Cache tensors use shape ``(batch, n_kv_heads, sequence, head_dim)``.
        """
        ## Step-0:- Declare the dimensions 
        batch_size, context_len, embd_dim = x.size()
        
        ## Check past Length 
        if cache is None:
            past_len =0 
        else:
            k_past, v_past = cache
            past_len = k_past.size(2)
        
        
        ## Step-1:- Build the Q, K , V matrices 
        Q, K, V = self.q_proj(x)  , self.k_proj(x),  self.v_proj(x)
        
        ## Step-2: - Prepare them for Multi head Attention 
        Q = Q.view(batch_size, context_len, self.config.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, context_len, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, context_len, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        
        ### Step-2(b) - Preparing KV Cache and Rope
        if self.config.use_rope:
            Q = self.rope(Q, position_offset = past_len)
            K = self.rope(K, position_offset = past_len)
            
        
        if cache is None:
            k_all = K
            v_all = V

        else:
            k_past, v_past = cache
            k_all = torch.cat([k_past, K],dim=2)
            v_all = torch.cat([v_past, V],dim=2)
        
        total_len = k_all.size(2)
        
        k_attn = k_all.repeat_interleave(self.n_rep,dim=1)
        v_attn = v_all.repeat_interleave(self.n_rep,dim=1)
        
        

        ## Step-3: - Dot Product of Q.KT and Divide by sqrt Dim Head for stability 
        # qkt = torch.einsum('bqhd, bkhd -> bhqk', Q, k_attn)
        qkt = Q @ k_attn.transpose(-2,-1)
        qkt = qkt * (self.head_dim ** -0.5)
        
        ### Step-3(b) - Causal mask apply with torch.tril
        mask = self.causal_mask[:, :, past_len:past_len+context_len, :total_len]
        qkt = qkt.masked_fill(mask==0, float("-inf"))
        
        ## Step-4: - Softmax 
        attn = F.softmax(qkt, dim=-1) 
        self.attn = attn 
        
        ## Step-5: - Multiply with V vector 
        # out = torch.einsum('bhqk, bhkd -> bqhd', attn, v_attn)
        out = attn @ v_attn

        
        
        ## Step-6:- Concat all heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, context_len, embd_dim)
        
        ## Step-7: - Pass through output proj and dropout layer
        out = self.o_proj(out) 
        out = self.dropout(out)
        
        new_cache = ((k_all, v_all) if use_cache else None)
        
        return out, new_cache






class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        
        assert config.n_heads % config.n_kv_heads == 0, "Number of query heads must be divisible by KV heads"
        
        
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        
        if self.config.use_rope:
            self.rope = RotaryPositionEmbedding(config.rope_config)
        
        self.dropout = nn.Dropout(config.dropout)
        
        
        self.n_rep = self.config.n_heads // self.config.n_kv_heads
        self.attn = None
        self.o_proj.NANOGPT_SCALE_INIT = 1
        
        
        
    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Return attention output and, when requested, the appended KV cache.

        Cache tensors use shape ``(batch, n_kv_heads, sequence, head_dim)``.
        """
        ## Step-0:- Declare the dimensions 
        batch_size, context_len, embd_dim = x.size()
        
        ## Check past Length 
        if cache is None:
            past_len =0 
        else:
            k_past, v_past = cache
            past_len = k_past.size(2)
        
        
        ## Step-1:- Build the Q, K , V matrices 
        Q, K, V = self.q_proj(x)  , self.k_proj(x),  self.v_proj(x)
        
        ## Step-2: - Prepare them for Multi head Attention 
        Q = Q.view(batch_size, context_len, self.config.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, context_len, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, context_len, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        
        ### Step-2(b) - Preparing KV Cache and Rope
        if self.config.use_rope:
            Q = self.rope(Q, position_offset = past_len)
            K = self.rope(K, position_offset = past_len)
            
        
        if cache is None:
            k_all = K
            v_all = V

        else:
            k_past, v_past = cache
            k_all = torch.cat([k_past, K],dim=2)
            v_all = torch.cat([v_past, V],dim=2)
        
        total_len = k_all.size(2)
        
        k_attn = k_all.repeat_interleave(self.n_rep,dim=1)
        v_attn = v_all.repeat_interleave(self.n_rep,dim=1)
        
        

        ## Step-3: - Dot Product of Q.KT and Divide by sqrt Dim Head for stability 
        # qkt = torch.einsum('bqhd, bkhd -> bhqk', Q, k_attn)
        qkt = Q @ k_attn.transpose(-2,-1)
        qkt = qkt * (self.head_dim ** -0.5)
        
        ### Step-3(b) - Causal mask apply with torch.tril
        # mask = self.causal_mask[:, :, past_len:past_len+context_len, :total_len]
        # qkt = qkt.masked_fill(mask==0, float("-inf"))
        
        ## Step-4: - Softmax 
        attn = F.softmax(qkt, dim=-1) 
        self.attn = attn 
        
        ## Step-5: - Multiply with V vector 
        # out = torch.einsum('bhqk, bhkd -> bqhd', attn, v_attn)
        out = attn @ v_attn

        
        
        ## Step-6:- Concat all heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, context_len, embd_dim)
        
        ## Step-7: - Pass through output proj and dropout layer
        out = self.o_proj(out) 
        out = self.dropout(out)
        
        new_cache = ((k_all, v_all) if use_cache else None)
        
        return out, new_cache
