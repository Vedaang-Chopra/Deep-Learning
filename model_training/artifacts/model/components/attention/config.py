
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rope:
    head_dim: int
    context_len: int
    theta: float = 10_000.0
    
    
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
    use_rope :bool = True
    rope_config: Rope = field(init=False)

    
    def __post_init__(self) -> None:
        self.validate()

        self.rope_config = Rope(
            head_dim=self.d_model // self.n_heads,
            context_len=self.max_seq_len,
            theta=self.rope_theta,
        )

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        head_dim = self.d_model // self.n_heads

        if head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")