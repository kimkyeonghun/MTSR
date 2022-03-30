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

class attn(nn.Module):
    #Bahdanau Attention
    def __init__(self, in_shape, out_shape):
        super(attn, self).__init__()
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
    def __init__(self):
        super(GraphAttentionLayer, self).__init__()
    