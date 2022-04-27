import copy
import math

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
    
    def forward(self, full, last, test):
        score = self.V(F.tanh(self.W1(last)+self.W2(full)))
        a_weight = F.softmax(score, dim=1)
        if test:
            context_vector = a_weight * full
            context_vector = torch.sum(context_vector, dim=1)
            score = score.squeeze(0).cpu().detach().numpy()
            return context_vector, score
            
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
        N = h.size()[0]

        #need to check
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc_layer, fc_layer):
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.fc_layer = fc_layer

    def calculate_attention(self, query, key, value, mask, test):
        d_k = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        attention_score = attention_score / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)

        if test:
            return attention_score

        attention_prob = F.softmax(attention_score, dim=-1)
        out = torch.matmul(attention_prob, value)

        return out

    def forward(self, query, key, value, test, mask=None):
        # query, key, value's shape: (n_batch, seq_len, d_embed) -> (1, 87, 64)
        # mask's shape: (n_batch, seq_len, seq_len) -> (1, 87, 87)
        n_batch = query.shape[0]

        def transform(x, fc_layer):
            # reshape (n_batch, seq_len, d_embed) to (n_batch, h, seq_len, d_k) -> (1, 87, 64) -> ()
            out = fc_layer(x) # out's shape (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)
            out = out.transpose(1,2)
            return out

        query = transform(query, self.query_fc_layer)
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)

        if mask is not None:
            mask = mask.unsqueeze(1)

        if test:
            attention_score = self.calculate_attention(query, key, value, mask, test)
            return attention_score

        out = self.calculate_attention(query, key, value, mask, test)
        out = out.transpose(1,2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        out = self.fc_layer(out)

        return out