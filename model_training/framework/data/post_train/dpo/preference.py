import torch 
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset



class PreferenceDataset(Dataset):
    
    def __init__(self, data, tokenizer):
        self.data = data
        
        self.encoded_texts =[]
        for entry in data:
            prompt = self.format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]
            
            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })
            
    
    def format_input(self, entry):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text
    
    
    
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    
    def __len__(self):
        return len(self.data)
    
    


def custom_collate_fn(
    batch, 
    pad_token_id = 505256,
    allowed_max_length= None, 
    mask_prompt_tokens=True,
    device = "cpu"
):
    # Initialize lists to hold batch data
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []

    }

    # Determine the longest sequence to set a common padding length
    max_length_common = 0
    
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max( len(item[key])+1  for item in batch)
            max_length_common = max(max_length_common, current_max)
    
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)
        
        for key in ["chosen", "rejected"]:
            # Adjust padding according to the common maximum length
            sequence = item[key]
            
            padded = sequence + [pad_token_id] * (max_length_common -len(sequence))
            
            mask[len(sequence):] = False