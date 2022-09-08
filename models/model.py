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
        self.text_time_lstm = [TimeLSTM(768, 64) for _ in range(stock_num)]
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
        self.attentions = SelfAttentionLayer(64, n_heads, nn.Linear(64, 64), nn.Linear(64, 64))
        #need to expand stock_num
        self.fc_layer = nn.Linear(64,2)
        
        #Price
        for i, p_g in enumerate(self.price_gru):
            self.add_module(f'price_gru{i}', p_g)
        for i, p_a in enumerate(self.price_attn):
            self.add_module(f'price_attn{i}', p_a)

        #Text
        for i, t_g in enumerate(self.text_time_lstm):
            self.add_module(f'text_time_lstm{i}', t_g)
        for i, t_a in enumerate(self.text_attn):
            self.add_module(f'text_attn{i}', t_a)
        for i, s_g in enumerate(self.seq_gru):
            self.add_module(f'seq_gru{i}', s_g)
        for i, s_a in enumerate(self.seq_attn):
            self.add_module(f'seq_attn{i}', s_a)

        #blending    
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
                y = self.text_time_lstm[i](text_input[i,j,:,:].reshape((1,num_text, 512)))
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
        
        feature = torch.Tensor((num_stock, 64))
        feature = torch.cat(rep)
        out_1 = F.tanh(self.blending[i](feature))
        x = F.dropout(feature, self.dropout, training = train)
        #self.info("Shape of output {}".format(x.shape))
        #self.info("Shape of adj {}".format(adj.shape))
        x = x.unsqueeze(0)
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

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size*4)
        self.U_all = nn.Linear(input_size, hidden_size*4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, _ = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()

        outputs = []
        hidden_state_h = []
        hidden_state_c = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c)) #Short-term memory
            c_s2 = c_s1 * timestamps[:,s:s+1].expand_as(c_s1) #Discounted shor-term memory
            c_1 = c - c_s1 #Long-term memory
            c_adj = c_1 + c_s2 #Adjusted previous memory
            outs = self.U_all(h) + self.W_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1) #forget, input, output, candidate
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            #c_tmp = torch.tanh(c_tmp) #Time-aware 논문은 이거
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(o)
            hidden_state_h.append(h)
            hidden_state_c.append(c)
        
        if reverse:
            outputs.reverse()
            hidden_state_h.reverse()
            hidden_state_c.reverse()

        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        #return outputs, (hidden_state_h, hidden_state_c)
        return outputs, (h, c)        

class Attention(nn.Module):
    def __init__(self, in_shape, maxlen=None, use_attention=True):
        super(Attention, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = nn.Linear(in_shape, in_shape)
            self.W2 = nn.Linear(in_shape, in_shape)
            self.V = nn.Linear(in_shape, 1)
        if maxlen != None:
            self.arange = torch.arange(maxlen)

    def forward(self, full, last, lens=None, dim=-1):
        if self.use_attention:
            score = self.V(F.tanh(self.W1(last) + self.W2(full)))
            attention_weights = F.softmax(score, dim=dim)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector
        else:
            return torch.mean(full, dim=dim)

class FISA(nn.Module):
    def __init__(self, num_stocks, logger):
        super(FISA, self).__init__()
        self.logger = logger
        #Text Encoder
        self.text_lstm = [nn.LSTM(768, 64) for _ in range(num_stocks)]
        for i, text_l in enumerate(self.text_lstm):
            self.add_module('textlstm{}'.format(i), text_l)
        
        self.time_lstm = [TimeLSTM(768, 64) for _ in range(num_stocks)]
        for i, time_l in enumerate(self.time_lstm):
            self.add_module('timelstm{}'.format(i), time_l)
        
        self.day_lstm = [nn.LSTM(64, 64) for _ in range(num_stocks)]
        for i, day_l in enumerate(self.day_lstm):
            self.add_module('daylstm{}'.format(i), day_l)

        self.text_attention = [Attention(64, 10) for _ in range(num_stocks)]
        for i, text_a in enumerate(self.text_attention):
            self.add_module('textattention{}'.format(i), text_a)

        self.day_attention = [Attention(64, 5) for _ in range(num_stocks)]
        for i, day_a in enumerate(self.day_attention):
            self.add_module('dayattention{}'.format(i), day_a)

        self.linear_stock = nn.Linear(64, 1)

    def info(self, text):
        self.logger.info(text)


    def forward(self, text_input, time_inputs, num_stocks):
        list_1 = []
        op_size = 64
        #text_input.size(0) = stock_num
        for i in range(text_input.size(0)):
            list_2 = []
            num_day = text_input.size(1)
            num_text = text_input.size(2)
            embed_dims = text_input.size(3)
            #intra Day Attention
            for j in range(num_day):
                y, (temp, _) = self.time_lstm[i](text_input[i, j, :, :].reshape(1, num_text, embed_dims), time_inputs[i,j,:].reshape(1, num_text))
                y = self.text_attention[i](y, temp, num_text)
                list_2.append(y)

            text_vectors = torch.Tensor((1, num_day, op_size))
            text_vectors = torch.cat(list_2)
            #inter-day attention
            text, (temp2, _ ) = self.day_lstm[i](text_vectors.reshape(1, num_day, op_size))
            text = self.day_attention[i](text, temp2, num_day)
            list_1.append(text.reshape(1, op_size))

        #FFNN, output
        ft_vec = torch.Tensor((num_stocks, op_size))
        ft_vec = torch.cat(list_1)
        op = F.leaky_relu(self.linear_stock(ft_vec))
        return op


