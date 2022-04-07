import random

import torch
from torch.utils.data import Dataset
import numpy as np

class MTSRDataset(Dataset):
    def __init__(self, features):
        self.items = features
        self.total_item = self.count()

    def __len__(self):
        return self.total_item

    def __getitem__(self, i):
        return 