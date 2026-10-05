import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataLoader(Dataset):
    def __init__(self, df, 
                train_column,
                label_column,
                tokenizer, 
                pad_token_id,  
                max_length = None):
        
        self.x = [tokenizer.encode(i) for i in df[train_column].values]
        self.y = [int(i) for i in df[label_column].values]
                
        if max_length is None:
            self.max_length = max(len(encoded_text) for encoded_text in self.x)
        else:
            self.max_length = max_length
            
            self.x = [
                i[:self.max_length] for i in self.x
            ]
        
        self.x = [
            i + [pad_token_id] * (self.max_length - len(i))
            for i in self.x
        ]

        
    def __len__(self):
        return len(self.x)
    
    
    def __getitem__(self, idx):
        encoded = torch.tensor(self.x[idx], dtype = torch.long)
        label = torch.tensor(self.y[idx], dtype = torch.long)
        return (
            encoded, label
        )

def return_dataloader(dataset, 
                    batch_size, 
                    shuffle, 
                    workers, 
                    drop_last):
    
    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=workers, 
        drop_last=drop_last
    )
    