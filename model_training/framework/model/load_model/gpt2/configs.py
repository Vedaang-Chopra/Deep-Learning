
from dataclasses import dataclass
    


### Setting up all Configurations 
DEFAULT_MODEL_TYPE = "gpt2-small"

BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

model_configs = {
            "gpt2-small":   dict(n_layer=12, n_head=12, emb_dim=768),
            "gpt2-medium":  dict(n_layer=24, n_head=16, emb_dim=1024),
            "gpt2-large":   dict(n_layer=36, n_head=20, emb_dim=1280),
            "gpt2-xl":      dict(n_layer=48, n_head=25, emb_dim=1600),
}

@dataclass
class GPT2Config:
    context_len:int = 1024
    vocab_size: int = 50257 # 50K BPE Merges, 256 Byte tokens, 1 |<endofttext>|
    n_layer:int = 12
    n_head:int = 12
    emb_dim:int = 768   
    
    input_prompt = "Every effort moves"



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



def load_model_config(dataset_max_len:int, model_type = DEFAULT_MODEL_TYPE):
    
    
    assert model_type in {'gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl' }
    
    print("loading wights from pretrained gpt: %s" % model_type)
    
    assert dataset_max_len <= BASE_CONFIG["context_length"], (
        f"Dataset length {dataset_max_len} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )
    
    config_args = model_configs[model_type]
    print(config_args)
    gpt2config = GPT2Config(**config_args)    
    
    return gpt2config
    