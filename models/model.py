import torch
import torch.nn as nn

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()

        self.price_gru = []
        self.price_attn = []
        self.text_gru = []
        self.text_attn = []
        self.seq_gru = []
        self.seq_gru = []
        self.bilinear = []

        #norm?

        self.blending = []

        self.attention = []

        self.out = []

    def forward():
        pass






