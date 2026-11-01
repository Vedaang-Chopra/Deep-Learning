
import torch
import torch.nn as nn
from typing import TypeAlias
from torch.nn import functional as F
from dataclasses import dataclass, field

from model_training.framework.model.components.other.rope import RotaryPositionEmbedding
from model_training.framework.model.components.other.normalization.rms_norm import RMSNorm

KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]

@dataclass(frozen=True)
class Rope:
    head_dim: int
    context_len: int
    theta: float

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        config.validate()
        
        ## Step-1: - Setup all Config needed for this class,  Define the attention configs
            
        self.config = config
        self.emd_dim = config.emb_dim
        self.n_kv_heads = config.n_kv_group
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qk_norm = config.qk_norm
        self.n_rep = self.config.n_heads // self.config.n_kv_heads
        self.use_rope = config.use_rope
        self.dtype = config.dtype
        self.max_seq_len = config.context_len
        
        q_width = self.n_heads * self.head_dim
        kv_width = self.n_kv_heads * self.head_dim
        
        # Calculate the head_dim based on the number of heads and emb_dim
        if self.head_dim is None:
            assert self.emd_dim % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            self.head_dim = config.emd_dim // config.n_heads
        else:
            self.head_dim = config.head_dim
        
        assert config.n_heads % config.n_kv_heads == 0, "Number of query heads must be divisible by KV heads"
        

        ## Step-2: -  Declare the QKV projection matrices
        self.q_proj = nn.Linear(self.emd_dim, q_width, bias=False, dtype=self.dtype)
        self.k_proj = nn.Linear(self.emd_dim, kv_width, bias=False, dtype=self.dtype)
        self.v_proj = nn.Linear(self.emd_dim, kv_width, bias=False, dtype=self.dtype)
        self.o_proj = nn.Linear(q_width, self.emd_dim, bias=False, dtype=self.dtype)
        
        
        ## Step-3:-  Setting up RMS Norm for Qwen3 Module (Q and K Matrices)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        ## Step-4: - Setting up Rope as a Flag
        self.rope = RotaryPositionEmbedding(config) if config.use_rope else None
        
        ## Step-5: - Additional Layers 
        self.dropout = nn.Dropout(config.dropout)
        
        
        ### Step-6: - Register Causal torch.tril operations 
        self.register_buffer("causal_mask", torch.tril(torch.ones(self.max_seq_len, 
                    self.max_seq_len).view(1, 1, self.max_seq_len, self.max_seq_len)
                    ), persistent = False)
        
        
        
        ## Step-7: - Additional Variable for initial assignment, and plots 
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
        
        total_len = past_len + context_len
        if total_len > self.config.context_len:
            raise ValueError("sequence plus cache exceeds context_len")

        ## Step-1:- Build the Q, K , V matrices 
        Q, K, V = self.q_proj(x)  , self.k_proj(x),  self.v_proj(x)
        
        ## Step-2: - Prepare them for Multi head Attention 
        Q = Q.view(batch_size, context_len, self.config.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, context_len, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, context_len, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        
        ### Step-2(b) - Preparing KV Cache and Rope
        if self.config.use_rope:
            q = self.rope(q, position_offset=past_len)
            k = self.rope(k, position_offset=past_len)

        
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
        out = out.view(batch_size, context_len, self.n_heads * self.head_dim)
        
        ## Step-7: - Pass through output proj and dropout layer
        out = self.o_proj(out) 
        out = self.dropout(out)
        
        new_cache = ((k_all, v_all) if use_cache else None)
        
        return out, new_cache






class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        config.validate()
        
        self.config = config
        self.head_dim = self.emd_dim // config.n_heads
        self.emd_dim = config.emd_dim
        
        assert config.n_heads % config.n_kv_heads == 0, "Number of query heads must be divisible by KV heads"
        
        
        self.q_proj = nn.Linear(self.emd_dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.emd_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.emd_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.emd_dim, self.emd_dim, bias=False)
        
        
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
