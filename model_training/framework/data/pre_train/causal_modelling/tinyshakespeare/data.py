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


