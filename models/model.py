import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import gru, attn, GraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, n_feature, n_hidden, dropout, alpha, n_heads, stock_num):
        super(GAT, self).__init__()
        self.dropout = dropout

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
        self.attentions = [GraphAttentionLayer(n_feature, n_hidden, dropout, alpha, concat=True) for _ in range(n_heads)]
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

    def forward(self, text_input, price_input, label, adj, train):
        #num_text=5, num_day=5, f_price=3, num_stock=87
        num_text = text_input.size(2)
        num_day = text_input.size(1)
        f_price = price_input.size(2)
        num_stock = price_input.size(0)

        rep = []
        for i in range(num_stock):
            x = self.price_gru[i](price_input[i, :, :].reshape((1,num_day, f_price)))
            x = self.price_attn[i](*x).reshape((1,64))
            one_day = []
            for j in range(text_input.size(1)):
                y = self.text_gru[i](text_input[i,j,:,:].reshape((1,num_text, 512)))
                y = self.text_attn[i](*y).reshape((1,64))
                one_day.append(y)

            #바로 init..?
            news_vector = torch.Tensor((1, num_day, 64))
            news_vector = torch.cat(one_day)
            text = self.seq_gru[i](news_vector.reshape((1,num_day,64)))
            text = self.seq_attn[i](*text).reshape((1,64))
            combined = F.tanh(self.bilinear[i](text, x).reshape((1,64)))
            rep.append(combined.reshape(1,64))
        
        feature = torch.Tensor((num_stock, 64))
        feature = torch.cat(rep)
        out_1 = F.tanh(self.blending[i](feature))
        x = F.dropout(feature, self.dropout, training = train)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training = train)
        x = F.elu(self.out(x, adj))

        output = x + out_1
        #need to train
        output = F.softmax(output, dim=1)
        loss_fct = nn.CrossEntropoyLoss()
        loss = loss_fct(label, output)

        return [loss, output]








