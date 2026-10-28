from pathlib import Path
import torch

def load_raw_data(root_dataset_dir:Path):
    DATASET_DIR = root_dataset_dir / "tinyshakespeare"
    with open( (DATASET_DIR /'input.txt').resolve(), 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


def tokenized_data(raw_data:str, tokenizer):
    return tokenizer.encode(raw_data)
    
    

def return_data(root_dataset_dir, tokenizer):
    text = load_raw_data(root_dataset_dir)
    enc_data = tokenized_data(text, tokenizer)
    
    return enc_data


def prepare_one_batch(encoded_data, batch, context_len):
    
    x= encoded_data[:-1].view(batch, context_len)
    y = encoded_data[1:].view(batch, context_len)
    
    return x, y


class DataLoaderLite:
    def __init__(self, batch_size, seq_len, tokenizer, dataset_dir) -> None:
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.enc_data = return_data(dataset_dir, tokenizer)
        self.enc_data = torch.tensor(self.enc_data)
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