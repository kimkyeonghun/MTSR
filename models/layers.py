import torch
import torch.nn as nn
import torch.nn.functional as F

class gru(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(gru, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first =True)
     
    def forward(self, inputs):
        out, h_n = self.gru(inputs)
        return out, h_n

class Attn(nn.Module):
    #Bahdanau Attention
    def __init__(self, in_shape, out_shape):
        super(Attn, self).__init__()
        self.W1 = nn.Linear(in_shape, out_shape)
        self.W2 = nn.Linear(in_shape, out_shape)
        self.V = nn.Linear(in_shape, 1)
    
    def forward(self, full, last):
        score = self.V(F.tanh(self.W1(last)+self.W2(full)))
        a_weight = F.softmax(score, dim=1)
        context_vector = a_weight * full
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        print(self.W.shape)
        N = h.size()[0]

        #need to check
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        print(a_input.shape)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        print(e.shape)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime