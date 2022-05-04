import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import gru, Attn, GraphAttentionLayer, SelfAttentionLayer

class GAT(nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, dropout, alpha, n_heads, stock_num, logger):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.logger = logger

        #price encoder
        self.price_gru = [gru(3, 64) for _ in range(stock_num)]
        self.price_attn = [Attn(64, 64) for _ in range(stock_num)]

        #text embedding
        self.text_gru = [gru(512, 64) for _ in range(stock_num)]
        self.text_attn = [Attn(64, 64) for _ in range(stock_num)]
        self.seq_gru = [gru(64, 64) for _ in range(stock_num)]
        self.seq_attn = [Attn(64, 64) for _ in range(stock_num)]

        #multimodal bledning
        self.bilinear = [nn.Bilinear(64, 64, 64) for _ in range(stock_num)]
        self.blending = [nn.Linear(64, 2) for _ in range(stock_num)]

        #GAT
        #self.attentions = [GraphAttentionLayer(n_feature, n_hidden, dropout, alpha, concat=True) for _ in range(n_heads)]
        self.out = GraphAttentionLayer(n_feature * n_hidden, n_class, dropout, alpha, concat=False)

        #Self
        self.attentions = [SelfAttentionLayer(64, n_heads, nn.Linear(64, 64), nn.Linear(64, 64)) for _ in range(stock_num)]
        #need to expand stock_num
        self.fc_layer = nn.Linear(64,2)

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

    def info(self, text):
        self.logger.info(text)

    def extract_attention_map(self, text_input, price_input, stocks):
        num_text = text_input.size(3)
        num_day = text_input.size(2)
        f_price = price_input.size(3)
        num_stock = price_input.size(1)

        rep = []
        price_input = price_input.squeeze(0)
        text_input = text_input.squeeze(0)
        price_attentions = dict()
        intra_day_attentions = dict()
        inter_day_attentions = dict()
        for i in range(num_stock):
            x = self.price_gru[i](price_input[i, :, :].reshape((1,num_day, f_price)))

            x, price_attention = self.price_attn[i](*x, True)
            price_attentions[stocks[i]] = price_attention
            x = x.reshape((1,64))
            one_day = []
            one_attention = []
            for j in range(num_day):
                y = self.text_gru[i](text_input[i,j,:,:].reshape((1,num_text, 512)))
                y, intra_attention = self.text_attn[i](*y, True)
                one_attention.append(intra_attention)
                y = y.reshape((1,64))
                one_day.append(y)
            intra_day_attentions[stocks[i]] = one_attention
            #바로 init..?
            news_vector = torch.Tensor((1, num_day, 64))
            news_vector = torch.cat(one_day)
            text = self.seq_gru[i](news_vector.reshape((1,num_day,64)))
            text, inter_attention = self.seq_attn[i](*text, True)
            inter_day_attentions[stocks[i]] = inter_attention
            text = text.reshape((1, 64))
            combined = F.tanh(self.bilinear[i](text, x).reshape((1,64)))
            rep.append(combined.reshape(1,64))

        feature = torch.Tensor((num_stock, 64))
        feature = torch.cat(rep)
        x = F.dropout(feature, self.dropout, training = False)
        x = x.unsqueeze(0)
        attention_score = torch.cat([att(x, x, x, True) for att in self.attentions], dim=1)
        attention_score = attention_score.squeeze(0).cpu().detach().numpy()
            
        return price_attentions, intra_day_attentions, inter_day_attentions, attention_score

    # def forward(self, text_input, price_input, label, adj, train):
    #     #num_text=5->1, num_day=5, f_price=3, num_stock=87
    #     num_text = text_input.size(2)
    #     num_day = text_input.size(1)
    #     f_price = price_input.size(2)
    #     num_stock = price_input.size(0)
    #     # self.info("# of text: {}, # of day: {}, price feature: {}, # of stock: {}".\
    #     #            format(num_text, num_day, f_price, num_stock))

    #     rep = []
    #     # self.info("Shape of Price input {}".format(price_input.shape))
    #     # self.info("Shape of Text input {}".format(text_input.shape))
    #     for i in range(num_stock):
    #         x = self.price_gru[i](price_input[i, :, :].reshape((1,num_day, f_price)))
    #         x = self.price_attn[i](*x, False).reshape((1,64))
    #         one_day = []
    #         for j in range(num_day):
    #             y = self.text_gru[i](text_input[i,j,:,:].reshape((1,num_text, 512)))
    #             y = self.text_attn[i](*y, False).reshape((1,64))
    #             one_day.append(y)

    #         #바로 init..?
    #         news_vector = torch.Tensor((1, num_day, 64))
    #         news_vector = torch.cat(one_day)
    #         text = self.seq_gru[i](news_vector.reshape((1,num_day,64)))
    #         text = self.seq_attn[i](*text, False).reshape((1,64))
    #         combined = F.tanh(self.bilinear[i](text, x).reshape((1,64)))
    #         rep.append(combined.reshape(1,64))
        
    #     feature = torch.Tensor((num_stock, 64))
    #     feature = torch.cat(rep)
    #     out_1 = F.tanh(self.blending[i](feature))
    #     x = F.dropout(feature, self.dropout, training = train)
    #     #self.info("Shape of output {}".format(x.shape))
    #     #self.info("Shape of adj {}".format(adj.shape))
    #     x = x.unsqueeze(0)
    #     x = torch.cat([att(x, x, x, False) for att in self.attentions], dim=1)
    #     x = F.dropout(x, self.dropout, training = train)
    #     x = x.squeeze(0)
    #     x = F.elu(self.fc_layer(x))

    #     output = x + out_1
    #     #need to train
    #     output = F.softmax(output, dim=1)
    #     loss_fct = nn.CrossEntropyLoss()
    #     label = label.squeeze(0)
    #     #self.info("Shape of output {}".format(output.shape))
    #     #self.info("Shape of label {}".format(label.shape))
    #     loss = loss_fct(output, label.long())

    #     return [loss, output]

    def forward(self, text_input, price_input, label, adj, train):
        #num_text=5->1, num_day=5, f_price=3, num_stock=87
        num_text = text_input.size(2)
        num_day = text_input.size(1)
        f_price = price_input.size(2)
        num_stock = price_input.size(0)
        # self.info("# of text: {}, # of day: {}, price feature: {}, # of stock: {}".\
        #            format(num_text, num_day, f_price, num_stock))

        rep = []
        # self.info("Shape of Price input {}".format(price_input.shape))
        # self.info("Shape of Text input {}".format(text_input.shape))
        outputs = []
        for i in range(num_stock):
            x = self.price_gru[i](price_input[i, :, :].reshape((1,num_day, f_price)))
            x = self.price_attn[i](*x, False).reshape((1,64))
            one_day = []
            for j in range(num_day):
                y = self.text_gru[i](text_input[i,j,:,:].reshape((1,num_text, 512)))
                y = self.text_attn[i](*y, False).reshape((1,64))
                one_day.append(y)

            #바로 init..?
            news_vector = torch.Tensor((1, num_day, 64))
            news_vector = torch.cat(one_day)
            text = self.seq_gru[i](news_vector.reshape((1,num_day,64)))
            text = self.seq_attn[i](*text, False).reshape((1,64))
            combined = F.tanh(self.bilinear[i](text, x).reshape((1,64)))
            rep.append(combined.reshape(1,64))
        
            #feature = torch.Tensor((1, 64))
            #feature = torch.cat(rep)
            out_1 = F.tanh(self.blending[i](combined))
            x = F.dropout(combined, self.dropout, training = train)
        #self.info("Shape of output {}".format(x.shape))
        #self.info("Shape of adj {}".format(adj.shape))
            x = x.unsqueeze(0)
            x = self.attentions[i](x, x, x,False)
            self.info("Shape of attention output {}".format(x.shape))
            self.info("Shape of blending output {}".format(out_1.shape))
            output = x + out_1
            assert False
        x = torch.cat([att(x, x, x, False) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training = train)
        x = x.squeeze(0)
        x = F.elu(self.fc_layer(x))

        output = x + out_1
        #need to train
        output = F.softmax(output, dim=1)
        loss_fct = nn.CrossEntropyLoss()
        label = label.squeeze(0)
        #self.info("Shape of output {}".format(output.shape))
        #self.info("Shape of label {}".format(label.shape))
        loss = loss_fct(output, label.long())

        return [loss, output]









