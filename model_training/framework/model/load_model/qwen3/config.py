
import torch
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Rope:
    head_dim: int
    context_len: int
    theta: float = 10_000.0
    
   
    
## Copied from Sebastian Code
@dataclass
class Qwen3ModelConfig:
    vocab_size : int  = 151936              # Vocabulary size
    context_len : int = 40_960              # Length originally used during training
    emb_dim: int = 1024                     # Embedding dimension
    n_heads: int = 16                       # Number of attention heads
    n_layers:int = 28                       # Number of layers
    hidden_dim:int = 3072                   # Size of intermediate dim in FeedForward
    head_dim: int | None = None             # Size of the heads in GQA        # Calculate in Attention always
    qk_norm:bool =True                      # Whether to normalize queries & keys in GQA
    n_kv_group:int = 8                      # Key-Value groups for GQA
    use_rope:bool = True                    # Use Rope as Bool parameter
    rope_theta:float = 1_000_000.0          # The base in RoPE's "theta"
    dtype:torch.dtype = torch.bfloat16     # Lower-precision dtype to reduce memory
    dropout: float = 0.0

    @property
    def rope_config(self) -> Rope:
        return Rope(
            head_dim=self.head_dim,
            context_len=self.context_len,
            theta=self.rope_theta,
        )

    def __post_init__(self) -> None:
        if self.n_heads ==0:
            raise ValueError("Number of Heads can't be 0")
                
        
        if self.head_dim is None:
            if self.emb_dim % self.n_heads != 0:
                raise ValueError("emb_dim must be divisible by n_heads")
            self.head_dim = self.emb_dim // self.n_heads
        self.validate()

    @property
    def theta(self) -> float:
        """Expose the name expected by RotaryPositionEmbedding."""
        return self.rope_theta
    
    # Add inside Qwen3ModelConfig:
    @property
    def n_kv_heads(self) -> int:
        """Return the number of key/value heads."""
        return self.n_kv_group

    def validate(self) -> None:
        """Check dimensions and settings used by attention."""
        if min(
            self.emb_dim,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.context_len,
        ) <= 0:
            raise ValueError("Attention dimensions must be positive")

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        if self.use_rope and self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even when RoPE is enabled")

        if self.use_rope and self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")