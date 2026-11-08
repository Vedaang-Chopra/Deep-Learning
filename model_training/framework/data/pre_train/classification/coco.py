
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import v2



def return_transform():
    return v2.Compose([
    v2.ToImage(), 
    v2.RandomResizedCrop((224, 224), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class CoCoViT(IterableDataset):
    
    def __init__(self, image_stream, labels_by_id, category_to_index, transform, category_ids):
        self.image_stream = image_stream
        self.labels_by_id = labels_by_id
        self.category_to_index = category_to_index
        self.transform = transform
        self.category_ids =category_ids
        
    def __iter__(self, ):
        for row in self.image_stream:
            category_ids = self.labels_by_id.get(row["image_id"])
            if category_ids is None:
                continue
        
        target = torch.zeros(80, dtype=torch.float32)
        for category_id in self.category_ids:
            target[self.category_to_index[category_id]] = 1.0
        
        image = self.transform(row["image"].convert("RGB"))
        yield image, target
