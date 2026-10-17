
from dataclasses import dataclass

@dataclass
class Rope:
    head_dim: int
    context_len:int 
    theta:int
    
    
@dataclass
class TransformerConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1536
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    dropout: float = 0.0

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head dimension must be even")

    use_rope :bool = True
    rope_config:Rope 




        
        
