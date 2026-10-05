from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from model_training.artifacts.model.components.block.gpt2_block import Block


class GPT(nn.Module):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        
        ### Module Dict -> Allows to index using Keys....
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            ### Module List -> Arrays of Blocks
            h= nn.ModuleList([
                Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
            
        ))
        
        ## Final Linear Layer -> Embedding to Vocabulary (Generates the Final Word)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    
    @classmethod
    def from_pretrained(self, load_config, model_type):
        """Loads pre trained weights from huggingface"""
        
        model = GPT(load_config)
        
        sd = model.state_dict()
        # print(sd.keys())
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def forward(self, idx, targets =None):
        B, T = idx.size()
        
        assert T<= self.config.block_size, f"Cannot forward on Sequence Length {T}, block size is only {self.config.block_size}"
        
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
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits
    