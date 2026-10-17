import torch
from torch.utils.data import Dataset




from typing import Any


class InstructionDataset(Dataset):
    
    def __init__(self, ds, tokenizer, prompt_style) -> None:
        super().__init__()
        self.data = ds
        
        
        
        ## The overall Pipeline Looks here as 
        # 1. Load the data and prepare the prompt format and encode data
        # 2. Padding per batch- Here we Here we are padding per batch, to save on tokens; as every batch can be very long. So 1 batch has max length 10 tokens, other can be 20 post padding
        # 3. We create list of target token id (shifted by-1); like pre-training 
        # 4. One Caveat here the last padding tokens we will not keep it end of text, we will replace them with -100  to exclude them from training loss. Replace the last padding tokens, padding tokens help us create balanced batches. Loss should not be dependent on batches, thus we add mask tokens
        
        ## Step-1: - Encoded data
        self.encoded_texts = []
        for entry in range(0, len(self.data)):
            full_text = prompt_style.complete_response(instruction=self.data[entry]['instruction'], 
                                                    in_text=self.data[entry]['input'], 
                                                    out_text=self.data[entry]['output'])
            
            self.encoded_texts.append(tokenizer.encode(full_text))
            
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        return self.encoded_texts[index]        


# We Write a custom collate function, when our batches are of different lengths 
def custom_collate_fn(
    batch, 
    pad_token_id,
    ignore_index=-100, 
    allowed_max_length=None,
    device="cpu"
):
    ### Step-2:- Padding per Batch
    batch_max_length_seq = max(len(item)+1 for item in batch )
    
    inputs_lst, targets_lst =[], []
    for item in batch:
        new_item = item.copy()
        
        #  Add an <|endoftext|> token
        new_item += [pad_token_id]
        
        padded = (
            new_item + [pad_token_id] *(batch_max_length_seq - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        
        ## Mask Helps ignore it in training data 
        ## Replace all but the first padding token in targets by ignore_index
        mask = targets == pad_token_id
        # print(mask, mask.shape)
        
        indices = torch.nonzero(mask).squeeze()
        # print(indices, indices.shape)
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
    
    
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    
    
    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)
        
    return inputs_tensor, targets_tensor

