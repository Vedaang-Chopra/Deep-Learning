
import torch
import torch.nn as nn
from torch.nn import functional as F



from model_training.framework.model.components.attention.self_attn import FlashCausalSelfAttention
from model_training.framework.model.components.ffn.gpt2_ffn import MLP



class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_eps)
        # self.attn = CausalSelfAttention(config)
        self.attn = FlashCausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        
        attn_in = self.attn_norm(x)
        
        attn_out, new_cache = self.attn(attn_in, cache, use_cache)
        
        h = x + attn_out
        
        ffn_in = self.ffn_norm(h)
        ffn_out = self.ffn(ffn_in)
        
        out = h + ffn_out
        
        return out, new_cache
       