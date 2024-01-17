import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
# -----------------------------4.3.1 attention-Enhanced Transformer Encoder----------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device, max_seq_len=150):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # :param src: [bs, src len, hid dim]
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model,device,parameters,dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.device = device
        self.city = parameters.city

        self.mat_bias = nn.Linear(2,1, bias=True)
        self.Add_transformer_ST_flag = parameters.Add_transformer_ST_flag
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, temporal_mat,dis_mat,mask):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k,temporal_mat,dis_mat, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k,temporal_mat,dis_mat,mask, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # temporal_mat bacht_size*seq_len*seq_len
        seq_len = temporal_mat.shape[1]
        # temporal_mat
        if self.city == 'Porto':
            batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + temporal_mat*100)

            batch_dis_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + dis_mat*100)
        if self.city == 'yancheng':
            batch_temporal_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + temporal_mat)

            batch_dis_mat = 1.0 / torch.log(
                    torch.exp(torch.tensor(1.0).to(self.device)) + dis_mat)
        all_mat = torch.stack([batch_temporal_mat.to(torch.float32),batch_dis_mat.to(torch.float32)], dim=-1) # B T T 3
        all_mat = self.mat_bias(all_mat).squeeze().unsqueeze(1)  # (B, 1, T, T)
        scores += all_mat   # (B, 1, T, T)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        return output

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model,device, parameters,heads=8, dropout=0.1):
        super().__init__()
    
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model,device,parameters)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_3 = Norm(d_model)

    def forward(self, x,temporal_mat,dis_mat,mask,norm=False):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, temporal_mat,dis_mat,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x if not norm else self.norm_3(x)

class Transformer(nn.Module):
    def __init__(self,d_model, N, device,parameters, heads=8):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,device,parameters,heads) for _ in range(N)
        ])
        self.norm = Norm(d_model)

    def forward(self, src,temporal_mat,dis_mat,mask3d=None):
        """
        :param src: [bs, src len, hid dim]
        :param src_timeids:  bs src len 1
        :param mask: [bs, src len, src len]
        :return: encoder hidden, refined graph
        """
        x = self.pe(src) # pos_embedding
        for i in range(self.N):
            x = self.layers[i](x,temporal_mat,dis_mat,mask3d)   # transformer
        return self.norm(x)