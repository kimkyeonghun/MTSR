import torch
import torch.nn as nn
from layers import gru, attn, GraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, n_heads, stock_num):
        super(GAT, self).__init__()

        #price encoder
        self.price_gru = [gru(3, 64) for _ in range(stock_num)]
        self.price_attn = [attn(64, 64) for _ in range(stock_num)]

        #text embedding
        self.text_gru = [gru(512, 64) for _ in range(stock_num)]
        self.text_attn = [attn(64, 64) for _ in range(stock_num)]
        self.seq_gru = [gru(64, 64) for _ in range(stock_num)]
        self.seq_attn = [attn(64, 64) for _ in range(stock_num)]

        #multimodal bledning
        self.bilinear = [nn.Bilinear(64, 64, 64) for _ in range(stock_num)]
        self.blending = [nn.Linear(64, 2) for _ in range(stock_num)]

        #GAT
        self.attentions = [GraphAttentionLayer() for _ in range(n_heads)]
        self.out = GraphAttentionLayer()

        for i, p_g in enumerate(self.price_gru):
            self.add_module(f'price_gru{i}', p_g)
        for i, p_a in enumerate(self.price_attn):
            self.add_module(f'price_attn{i}', p_a)
        for i, t_g in enumerate(self.text_gru):
            self.add_module(f'text_gru{i}', t_g)
        for i, t_a in enumerate(self.text_attn):
            self.add_module(f'text_attn{i}', t_a)
        for i, s_g in enumerate(self.seq_gru):
            self.add_module(f'seq_gru{i}', s_g)
        for i, s_a in enumerate(self.seq_attn):
            self.add_module(f'seq_attn{i}', s_a)
        for i, bi in enumerate(self.bilinear):
            self.add_module(f'bilinear{i}', bi)
        for i, bl in enumerate(self.blending):
            self.add_module(f'blending{i}', bl)
        for i, attn in enumerate(self.attentions):
            self.add_module(f'attentions{i}', attn)

    def forward():
        pass






