import torch
from torch.utils.data import Dataset


DEVICE = torch.device("cuda")

class MTSRDataset(Dataset):
    def __init__(self, features):
        self.items = features
        self.total_item = self.count()

    def __len__(self):
        return self.total_item

    def count(self):
        return len(self.items)

    def __getitem__(self, i):
        price = self.items[i][0]
        text = self.items[i][1]
        label = self.items[i][2]
        return torch.tensor(price, dtype=torch.float32, device=DEVICE),\
         torch.tensor(text, dtype=torch.float32, device=DEVICE),\
         torch.tensor(label, dtype=torch.int64, device=DEVICE)