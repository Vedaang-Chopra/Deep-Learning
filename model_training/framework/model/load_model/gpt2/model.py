from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel

from model_training.framework.model.components.attention.self_attn import FlashCausalSelfAttention
from model_training.framework.model.components.ffn.gpt2_ffn import MLP



class Block(nn.Module):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = FlashCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)
        self.mlp = MLP (config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x    
    



class GPT(nn.Module):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        
        ### Module Dict -> Allows to index using Keys....
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.emb_dim),
            wpe = nn.Embedding(config.context_len, config.emb_dim),
            ### Module List -> Arrays of Blocks
            h= nn.ModuleList([
                Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.emb_dim)
            
        ))
        
        ## Final Linear Layer -> Embedding to Vocabulary (Generates the Final Word)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        ## Apply weight Initialization rules as per paper
        self.apply(self.__init_weights)
        
        ## The Final Linear is same as Embedding Layer -> Weight Sharing Scheme
        self.transformer.wte.weight = self.lm_head.weight
    
    def use_pretrained(self, hf_model: nn.Module) -> "GPT":
        """Copy a compatible Hugging Face GPT-2 model, transposing Conv1D weights."""
        hf_config = hf_model.config
        expected = (self.config.vocab_size, self.config.context_len, self.config.emb_dim,
                    self.config.n_layer, self.config.n_head)
        actual = (hf_config.vocab_size, hf_config.n_positions, hf_config.n_embd,
                  hf_config.n_layer, hf_config.n_head)
        if actual != expected:
            raise ValueError(f"GPT-2 configuration mismatch: expected {expected}, got {actual}")
        if not hf_config.tie_word_embeddings:
            raise ValueError("GPT-2 requires tied token embeddings and output weights")
        source = hf_model.state_dict()
        target = self.state_dict()
        names = [name for name in target if not name.endswith(".attn.bias")]
        source_names = {name for name in source if not name.endswith((".attn.bias", ".attn.masked_bias"))}
        if set(names) != source_names:
            raise ValueError(f"GPT-2 state keys differ: missing={set(names) - source_names}, unexpected={source_names - set(names)}")
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        with torch.no_grad():
            for name in names:
                value = source[name].t() if name.endswith(transposed) else source[name]
                if value.shape != target[name].shape:
                    raise ValueError(f"shape mismatch for {name}: {value.shape} != {target[name].shape}")
                target[name].copy_(value)
        return self

    @classmethod
    def from_pretrained(cls, load_config, model_type):
        """Load a named Hugging Face GPT-2 checkpoint into a matching configuration."""
        model_type = {"gpt2-small": "gpt2"}.get(model_type, model_type)
        if model_type not in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}:
            raise ValueError(f"Unsupported GPT-2 checkpoint: {model_type}")
        return cls(load_config).use_pretrained(GPT2LMHeadModel.from_pretrained(model_type))

    def forward(self, idx, targets =None):
        B, T = idx.size()
        
        assert T<= self.config.context_len, f"Cannot forward on Sequence Length {T}, block size is only {self.config.context_len}"
        
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device)
        pos_embd = self.transformer.wpe(pos)
        tok_embd = self.transformer.wte(idx)
        
        x = tok_embd + pos_embd
        
        ## Forward the blocks 
        for block in self.transformer.h:
            x = block(x)
        x= self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        ### Calculating the Loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits
    
            
    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std =0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) ## Xavier initialization is almost same as this
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            


def return_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model