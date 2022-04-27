import torch
from torch.utils.data import Dataset

import numpy as np

DEVICE = torch.device("cuda")

def collate(samples):
    #for batch
    prices = [None] * len(samples)
    texts = [None] * len(samples)
    labels = [None] * len(samples)
    dates = [None] * len(samples)
    stocks = [None] * len(samples)

    for i, (price, text, label, string) in enumerate(samples):
        prices[i] = price
        texts[i] = text
        labels[i] = label
        dates[i] = string['date']
        stocks[i] = string['stocks']
    
    return torch.tensor(price, dtype=torch.float32, device=DEVICE),\
         torch.tensor(text, dtype=torch.float32, device=DEVICE),\
         torch.tensor(label, dtype=torch.int64, device=DEVICE), dates, stocks

class MTSRDataset(Dataset):
    def __init__(self, features):
        self.items = features
        self.total_item = self.count()

    def __len__(self):
        return self.total_item

    def count(self):
        return len(self.items)

    def __getitem__(self, i):
        text = self.items[i][0][0]
        price = self.items[i][0][1]
        label = self.items[i][0][2]
        date = self.items[i][0][3][0]
        print("text", text.shape)
        
        stocks = self.items[i][1]
        strings = dict()
        strings['date'] = date
        strings['stocks'] = stocks

        return torch.tensor(price, dtype=torch.float32, device=DEVICE),\
         torch.tensor(text, dtype=torch.float32, device=DEVICE),\
         torch.tensor(label, dtype=torch.int64, device=DEVICE), strings