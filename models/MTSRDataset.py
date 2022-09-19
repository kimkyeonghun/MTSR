import torch
from torch.utils.data import Dataset

DEVICE = torch.device("cuda")


def collate(samples):
    # for batch
    prices = [None] * len(samples)
    texts = [None] * len(samples)
    times = [None] * len(samples)
    labels = [None] * len(samples)
    dates = [None] * len(samples)
    stocks = [None] * len(samples)

    for i, (price, text, time, label, string) in enumerate(samples):
        prices[i] = price.cpu().detach().numpy()
        texts[i] = text.cpu().detach().numpy()
        times[i] = time.cpu().detach().numpy()
        labels[i] = label.cpu().detach().numpy()
        dates[i] = string['date']
        stocks[i] = string['stocks']

    return torch.tensor(prices, dtype=torch.float32),\
        torch.tensor(texts, dtype=torch.float32),\
        torch.tensor(times, dtype=torch.float32),\
        torch.tensor(labels, dtype=torch.float32), dates, stocks


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
        time = self.items[i][0][2]
        label = self.items[i][0][3]
        date = self.items[i][0][4][0]

        stocks = self.items[i][1]
        strings = dict()
        strings['date'] = date
        strings['stocks'] = stocks

        return torch.tensor(price, dtype=torch.float32),\
            torch.tensor(text, dtype=torch.float32),\
            torch.tensor(time, dtype=torch.float32),\
            torch.tensor(label, dtype=torch.float32), strings
