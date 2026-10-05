import torch
from pathlib import Path
import os
### Dataset: - Allows you to load data into Tensors
### DataLoader: - Wraps an Iterable around Dataset
from torch.utils.data import DataLoader, Dataset

### For image Transforms: - 
from torchvision.transforms import v2


class PetsDataset(Dataset):
    '''
    This is a class trying to load the PetsDataset for training.
    '''
    
    def __init__(self, path_to_dataset_root:Path):
        self.data_root =path_to_dataset_root
        self.classes ={
            'cat' : 0,
            'dog' :1
        }
        print(os.listdir(self.data_root))
    
    
    