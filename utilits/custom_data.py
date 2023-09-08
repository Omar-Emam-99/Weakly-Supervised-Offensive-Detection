"""CustomDatasets to suits Dataloader"""
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    
    def __init__(self, encodings):
        
        self.encodings = encodings
        
    def __getitem__(self, idx):
        
        input_ids = self.encodings.input_ids[idx]
        attention_mask = self.encodings.attention_mask[idx]
        token_type_ids = self.encodings.token_type_ids[idx]

        return {
            "input_ids" : input_ids,
            "attention_mask" :attention_mask ,
            "token_type_ids" :  token_type_ids
        }

    def __len__(self):
        return len(self.encodings)