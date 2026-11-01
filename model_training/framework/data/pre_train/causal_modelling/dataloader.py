from dataclasses import dataclass
import torch
from model_training.framework.data.pre_train.causal_modelling.tinyshakespeare.data import prepare_one_batch

@dataclass
class CausalLanguageModellingDataset:
    total_batch_size:int = 1e6
    micro_batch_size:int = 8
    seq_len:int = 1024







class DataLoaderLite:
    def __init__(self, enc_data, batch_size, seq_len, tokenizer) -> None:
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.enc_data = torch.tensor(enc_data)
        if len(self.enc_data) < batch_size * seq_len + 1:
            raise ValueError("encoded data must contain at least batch_size * seq_len + 1 tokens")
        print(f"Loaded {len(self.enc_data)} tokens")
        print(f"1 epoch = {len(self.enc_data) // (self.batch_size * self.seq_len)} tokens")
        
        ## 
        self.current_position =0
        
    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        batch_tokens = self.enc_data[self.current_position: self.current_position + B*T + 1]
        
        x, y = prepare_one_batch(batch_tokens, B, T)
        
        self.current_position += B*T
        
        if self.current_position + (B*T +1) > len(self.enc_data):
            self.current_position = 0
        
        return x, y
