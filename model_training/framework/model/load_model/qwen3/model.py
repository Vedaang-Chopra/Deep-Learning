import torch 
import torch.nn as nn
from pathlib import Path

from model_training.framework.model.load_model.qwen3.config import Qwen3ModelConfig
from transformers import AutoModelForCausalLM

from model_training.framework.model.components.attention.mha import CausalMultiHeadAttention
from model_training.framework.model.components.ffn.qwen3_ffn import Qwen3FFn
from model_training.framework.model.components.other.normalization.rms_norm import RMSNorm

class Qwen3Block(nn.Module):
    
    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        
        ## Step-1: - Setup all Config needed for this class
        self.emb_dim = config.emb_dim 
        self.n_layers = config.n_layers
        
    
        ## Step-2: - Setup all layers 
        # 1. Input Pre Layer Norm 
        self.input_layernorm = RMSNorm(dim= self.emb_dim)
        
        # 2. Grouped Query Attention
        self.self_attn = CausalMultiHeadAttention(config)
        
        # 3. 2nd Layer Norm 
        self.post_attention_layernorm = RMSNorm(dim= self.emb_dim)
        
        # 4. Feed Forward Layer
        self.mlp = Qwen3FFn(config)
        
        
        
    
    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Apply one transformer block and return its optional KV cache."""
        residual = x
        x = self.input_layernorm(x)
        x, next_cache = self.self_attn(x, cache=cache, use_cache=use_cache)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x, next_cache
        



class Qwen3Model(nn.Module):
    
    
    def __init__(self, config: Qwen3ModelConfig):
        super().__init__()
        
        
        ## Step-1: - Setup all Config needed for this class
        
        self.emb_dim = config.emb_dim 
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.dtype = config.dtype
        
        ## Step-2: - Setup all layers 
        
        # 1. Tokenization layer
        self.embed_tokens = nn.Embedding(self.vocab_size, self.emb_dim, dtype = self.dtype)
        
        # 2. Repeating Block 
        self.layers = nn.ModuleList(
            [Qwen3Block(config) for _ in range(self.n_layers)]
        )
        
        
        # 3. Final Layer Norm
        self.norm = RMSNorm(dim= self.emb_dim)
        
        # 4. Final Linear Layer
        self.lm_head = nn.Linear(self.emb_dim , self.vocab_size, bias=False, dtype = self.dtype)
        
        self.lm_head.weight = self.embed_tokens.weight
    
    def forward(
        self,
        in_tokens: torch.Tensor,
        cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | None]:
        """Return logits and an optional KV cache for each block."""
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache must contain one entry per layer")

        x = self.embed_tokens(in_tokens)
        next_caches = []

        for i, block in enumerate(self.layers):
            block_cache = None if cache is None else cache[i]
            x, next_cache = block(x, cache=block_cache, use_cache=use_cache)

            if use_cache:
                next_caches.append(next_cache)

        logits = self.lm_head(self.norm(x))
        return logits, tuple(next_caches) if use_cache else None

    def use_pretrained(self, hf_model: nn.Module) -> "Qwen3Model":
        """Load weights from a compatible Hugging Face Qwen3 model."""
        if not hf_model.config.tie_word_embeddings:
            raise ValueError("This model currently requires tied embeddings")

        weights = {
            name.removeprefix("model."): tensor
            for name, tensor in hf_model.state_dict().items()
        }

        if "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["embed_tokens.weight"]

        self.load_state_dict(weights, strict=True)
        return self
            
        
        



