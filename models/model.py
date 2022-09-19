import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size*4).to("cuda:1")
        self.U_all = nn.Linear(input_size, hidden_size*4).to("cuda:1")
        self.W_d = nn.Linear(hidden_size, hidden_size).to("cuda:1")
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, _ = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        if self.cuda_flag:
            h = h.to("cuda:1")
            c = c.to("cuda:1")

        outputs = []
        hidden_state_h = []
        hidden_state_c = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))  # Short-term memory
            # Discounted shor-term memory
            c_s2 = c_s1 * timestamps[:, s:s+1].expand_as(c_s1)
            c_1 = c - c_s1  # Long-term memory
            c_adj = c_1 + c_s2  # Adjusted previous memory
            #outs: [b, hid*4]
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            # forget, input, output, candidate
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.tanh(c_tmp)  # Time-aware 논문은 이거
            #c_tmp = torch.sigmoid(c_tmp)
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

        # return outputs, (hidden_state_h, hidden_state_c)
        return outputs, (h, c)


class Attention(nn.Module):
    def __init__(self, in_shape, maxlen=None, use_attention=True, device='cuda:0'):
        super(Attention, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = nn.Linear(in_shape, in_shape).to(device)
            self.W2 = nn.Linear(in_shape, in_shape).to(device)
            self.V = nn.Linear(in_shape, 1).to(device)
        if maxlen != None:
            self.arange = torch.arange(maxlen)

    def forward(self, full, last, lens=None, dim=1):
        if self.use_attention:
            score = self.V(F.tanh(self.W1(last) + self.W2(full)))
            attention_weights = F.softmax(score, dim=dim)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector
        else:
            return torch.mean(full, dim=dim)


class FISA(nn.Module):
    def __init__(self, num_stocks, logger=False):
        super(FISA, self).__init__()
        self.logger = logger

        # Text Encoder
        #self.text_lstm = [nn.DataParallel(nn.LSTM(768, 64), device_ids=[0, 1]) for _ in range(num_stocks)]
        # self.text_lstm = [nn.LSTM(768, 64) for _ in range(num_stocks)]
        # for i, text_l in enumerate(self.text_lstm):
        #     self.add_module('textlstm{}'.format(i), text_l)

        #self.time_lstm = [nn.DataParallel(TimeLSTM(768, 64, cuda_flag=True), device_ids=[0, 1]) for _ in range(num_stocks)]
        self.time_lstm = [TimeLSTM(768, 64, cuda_flag=True).to(
            "cuda:1") for _ in range(num_stocks)]
        for i, time_l in enumerate(self.time_lstm):
            self.add_module('timelstm{}'.format(i), time_l)

        #self.day_lstm = [nn.DataParallel(nn.LSTM(64, 64), device_ids=[0, 1]) for _ in range(num_stocks)]
        self.day_lstm = [nn.LSTM(64, 64).to("cuda:1")
                         for _ in range(num_stocks)]
        for i, day_l in enumerate(self.day_lstm):
            self.add_module('daylstm{}'.format(i), day_l)

        #self.text_attention = [nn.DataParallel(Attention(64, 10), device_ids=[0, 1]) for _ in range(num_stocks)]
        self.text_attention = [
            Attention(64, 10, device='cuda:1') for _ in range(num_stocks)]
        for i, text_a in enumerate(self.text_attention):
            self.add_module('textattention{}'.format(i), text_a)

        #self.day_attention = [nn.DataParallel(Attention(64, 5), device_ids=[0, 1]) for _ in range(num_stocks)]
        self.day_attention = [Attention(64, 5, device='cuda:1')
                              for _ in range(num_stocks)]
        for i, day_a in enumerate(self.day_attention):
            self.add_module('dayattention{}'.format(i), day_a)

        #self.linear_stock = nn.DataParallel(nn.Linear(64, 1), device_ids=[0, 1])
        self.linear_stock = nn.Linear(64, 1).to("cuda:1")
        self.leaky_relu = F.leaky_relu
        # self.linear_stock = [nn.Linear(64, 1).to("cuda:0") for _ in range(num_stocks)]
        # for i, ls in enumerate(self.linear_stock):
        #     self.add_module('FFNN{}'.format(i), ls)

    def info(self, text):
        self.logger.info(text)

    def print_layer_parameter(self,):
        pass

    # def forward(self, text_input, time_inputs, num_stocks):
    #     list_1 = []
    #     op_size = 64
    #     #text_input.size(0) = stock_num
    #     #text_input = [stock_num, n_day, n_text, n_seq], [b, 87, 5, 10, 60]
    #     for i in range(text_input.size(0)):
    #         list_2 = []
    #         num_day = text_input.size(1)
    #         num_text = text_input.size(2)
    #         n_seq = text_input.size(3)
    #         #intra Day Attention
    #         for j in range(num_day):
    #             #encoded_input = self.bert(text_input[i, j, :].to("cuda:1"))['last_hidden_state'].to("cuda:0")
    #             encoded_input = self.bert.embeddings(text_input[i, j, :].to("cuda:1")).to("cuda:0")
    #             encoded_input = torch.sum(encoded_input,1)/n_seq
    #             self.x = encoded_input
    #             y, (temp, _) = self.time_lstm[0](encoded_input.reshape(1, num_text, 768), time_inputs[i,j,:].reshape(1, num_text).to("cuda:0"))
    #             self.y1 = y
    #             y = self.text_attention[0](y.to("cuda:0"), y.to("cuda:0"), num_text)
    #             self.y2 = y
    #             list_2.append(y)
    #             del encoded_input
    #             del y
    #         text_vectors = torch.cat(list_2)
    #         #inter-day attention
    #         text, (temp2, _ ) = self.day_lstm[0](text_vectors.reshape(1, num_day, op_size))
    #         text = self.day_attention[0](text, text, num_day)
    #         self.text = text
    #         list_1.append(text.reshape(1, op_size))
    #         #print(text)
    #         # op = 3*torch.tanh(self.linear_stock[num_stocks+i](text))
    #         # list_1.append(op)
    #     #FFNN, output
    #     # ft_vec = torch.Tensor((text_input.size(0), op_size))
    #     ft_vec = torch.cat(list_1)
    #     #op = F.leaky_relu(self.linear_stock(ft_vec))
    #     op = 3*torch.tanh(self.linear_stock(ft_vec))
    #     return op

    def forward(self, text_input, time_inputs, num_layer):
        list_1 = []
        op_size = 64
        # text_input.size(0) = stock_num
        #text_input = [stock_num, n_day, n_text, n_seq], [b, 87, 5, 10, 60]
        i = num_layer
        list_2 = []
        num_day = text_input.size(1)
        num_text = text_input.size(2)
        # intra Day Attention
        for j in range(num_day):
            y, (temp, _) = self.time_lstm[i](text_input[0, j, :].reshape(1, num_text, 768).to(
                "cuda:1"), time_inputs[0, j, :].reshape(1, num_text).to("cuda:1"))
            #y = self.leaky_relu(y)
            y = torch.tanh(y)
            y = self.text_attention[i](y, temp, num_text)
            list_2.append(y)
            del y
        text_vectors = torch.Tensor((1, num_day, op_size))
        text_vectors = torch.cat(list_2)
        # inter-day attention
        text, (temp2, _) = self.day_lstm[i](
            text_vectors.reshape(1, num_day, op_size))
        text = torch.tanh(text)
        text = self.day_attention[i](text, temp2, num_day)
        list_1.append(text.reshape(1, op_size))

        #FFNN, output
        ft_vec = torch.Tensor((1, op_size))
        ft_vec = torch.cat(list_1)
        #op = F.leaky_relu(self.linear_stock(ft_vec))
        op = 3*torch.tanh(self.linear_stock(ft_vec))
        return op
