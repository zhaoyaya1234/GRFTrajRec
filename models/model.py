import random
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from models.model_utils import get_dict_info_batch,get_grid_dict
from models.model_utils import rate2gps
from models.model_utils import get_gps_distance,get_ID_distance,get_road_prob_minus
from models.model_utils import cal_temporal_mat,cal_dis_mat
from models.transformer_layer import  Transformer
import datetime

def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        pred = x_exp / x_exp_sum
        pred = torch.clip(pred, 1e-6, 1)
        output_custom = torch.log(pred)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom

class Extra_MLP(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.pro_input_dim = parameters.pro_input_dim -1 
        self.pro_output_dim = parameters.pro_output_dim
        self.fc_out = nn.Linear(self.pro_input_dim, self.pro_output_dim)
        self.embedding = nn.Embedding(parameters.uid_num,self.pro_output_dim)
    def forward(self, x): # x 1 uid + 24 hour + 1 holiday
        out = self.embedding(x[:,0].long())+torch.tanh(self.fc_out(x[:,1:]))
        return out

# ------------------------4.1   Trajectory-Aware Graph Representation  -------------------------------------
# ------------------------GeoGAT-----------------------------------
class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_feats, h_feats, cached=True)
        self.conv2 = GATConv(h_feats, num_classes, cached=True)

    def forward(self, x,edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # 使用第二个GCNConv
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x)
        # (id_size, embedding_size)
        return x

class RoadGNN(torch.nn.Module):
    def __init__(self, parameters,alpha = 0.2):
        super(RoadGNN, self).__init__()
        self.params = parameters
        self.device = parameters.device
        self.hidden_dim = parameters.hidden_dim
        self.hid_dim = parameters.hid_dim
        self.GAT = GAT(self.hidden_dim,self.hidden_dim,self.hid_dim)
        self.fc3 = nn.Linear(2*self.hid_dim,self.hid_dim)

    def forward(self,road_network): 
        x, edge_index = road_network.x, road_network.edge_index
        node_embeddings_vec = self.GAT(x,edge_index)
        return node_embeddings_vec
# ---------------------ablation study change CandiGNN to the follwing GCN---------------
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        
        # 定义第一个GCN卷积层
        self.conv1 = GCNConv(in_feats, h_feats)
        # 定义第二个GCN卷积层
        self.conv2 = GCNConv(h_feats, num_classes)

    def forward(self, x, edge_index):
        # 通过第一个GCN卷积层并应用ReLU激活函数
        x = F.relu(self.conv1(x, edge_index))
        # 通过第二个GCN卷积层并应用ReLU激活函数
        x = F.relu(self.conv2(x, edge_index))
        # 应用Dropout
        x = F.dropout(x, training=self.training)
        return x  
# ---------------------------------------CandiGNN---------------------------------
class ProbGAT(torch.nn.Module):
    def __init__(self, parameters,alpha = 0.2):
        super(ProbGAT, self).__init__()

        self.hidden_dim = parameters.hidden_dim
        self.hid_dim =  parameters.hid_dim
        self.device = parameters.device
        self.emb_id = nn.Parameter(torch.rand(parameters.id_size, parameters.hidden_dim))
        self.params = parameters
        self.Add_change_GNN_flag= parameters.Add_change_GNN_flag
        if self.Add_change_GNN_flag:
            self.GCN = GCN(self.hidden_dim,self.hidden_dim,self.hid_dim)
        else:
            self.neighbor_all = parameters.neighbor_all
            self.att_fc1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            self.att_fc2 = torch.nn.Linear(self.hidden_dim, 1)
            self.a = torch.nn.Parameter(torch.randn(2*self.hidden_dim,1))
            self.alpha = alpha
            self.w = torch.nn.Parameter(torch.randn(2, self.hidden_dim, self.hidden_dim))
            self.fc1 = torch.nn.Linear(self.hidden_dim,self.hidden_dim)
            self.fc2 = torch.nn.Linear(self.hidden_dim, parameters.hid_dim)
    def forward(self,u): 
        # edge_index  left-> right
        # 1 get attention
        x = self.emb_id
        edge_index = self.params.road_network.edge_index
        if self.Add_change_GNN_flag:
            x = self.GCN(x,edge_index)
        else:
            k, i = edge_index[0],edge_index[1] # k->i
            h = (u[k]-u[i]) * (x[k] - x[i]) 
            h = torch.relu(self.att_fc1(h))
            alpha= torch.softmax(self.att_fc2(h), dim=0) # 列平均
            # 2  get neighbor emb
            zeros = torch.zeros(1,self.hidden_dim).to(self.device) 
            neighbor_emb  = torch.cat([x[k]*alpha, zeros], dim=0)
            x = x@ self.w[0] + torch.sum(neighbor_emb[self.neighbor_all[range(len(x))]],dim=0)@ (self.w[1])  # 行平均
            x = torch.relu(self.fc1(x))
            x= self.fc2(x)
        return x
# ------------------------GET emb -------------------------------------
class Get_Graph_Emb(nn.Module):
    def __init__(self, parameters):
        super(Get_Graph_Emb, self).__init__()
        self.hid_dim = parameters.hid_dim
        self.device = parameters.device

    def get_emb(self, road_emb, src_rids):
        embedded_seq_tensor = torch.zeros((src_rids.shape[0], src_rids.shape[1], self.hid_dim),
                                          dtype=torch.float32)
        for idx in range(len(src_rids)):
            embedded_seq_tensor[idx, :] = road_emb.index_select(0, src_rids[idx])
        return embedded_seq_tensor
    
    def forward(self,node_embeddings,src_rids):
        # src_rids seq_len*batch_size*1
        bacth_size = src_rids.shape[1]
        src_rids = src_rids.permute(1,0,2).squeeze()
        cons_emb = self.get_emb(node_embeddings, src_rids).to(self.device)  # batch*seq_len*embedding_size
        return cons_emb
# ------------------------graph fusion  -------------------------------------  
class Co_Att(nn.Module):
    def __init__(self, dim):
        super(Co_Att, self).__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.temperature = dim ** 0.5
        self.FFN = nn.Sequential(
            nn.Linear(dim, int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_s, seq_t):
        h = torch.stack([seq_s, seq_t], 2)  # # torch.Size([batch_size,seq_len, 2, hid_dim])
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)

        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)

        att_s = attn_o[:, :, 0, :]
        att_t = attn_o[:, :, 1, :]

        return att_s, att_t
# ------------------------ 4.2 Spatiotemporal Trajectory Representatio  -------------------------------------
# ------------------------ 4.2.1 time emb -------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, parameters, device):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.hidden_dim = 8
        # 使用一个简单的线性层来实现连续时间嵌入
        self.time_linear = nn.Linear(1, parameters.time_dim)

    def forward(self, time_seqs, seq_lengths):
        all_list = []
        for one_seq in time_seqs:
            # 转换时间戳为一天中的秒数
            seconds_of_day = []
            for timestamp in one_seq:
                t = datetime.datetime.fromtimestamp(timestamp)
                second = t.hour * 3600 + t.minute * 60 + t.second
                seconds_of_day.append(second)

            # 将秒数转换为张量，并进行归一化
            seconds_of_day = torch.tensor(seconds_of_day).float().unsqueeze(1)
            normalized_time = (seconds_of_day - seconds_of_day.min()) / (seconds_of_day.max() - seconds_of_day.min())
            embed = self.time_linear(normalized_time.to(self.device))
            all_list.append(embed)

        # 对齐并堆叠所有序列
        max_len = max(seq_lengths)
        embedded_seq_tensor = torch.zeros((len(all_list), max_len, self.hidden_dim), dtype=torch.float32).to(self.device)
        for i, embed in enumerate(all_list):
            length = seq_lengths[i]
            embedded_seq_tensor[i, :length] = embed[:length]

        return embedded_seq_tensor
# ------------------------4.2.2 spatiial position encoding -------------------------------------
import math
class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim, max_len=100):
        super(PositionalEncoding, self).__init__()

        # 创建一个位置信息矩阵
        pe = torch.zeros(max_len, pos_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / pos_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 从不需要梯度计算的变量中删除pe，更多详情参见文档
        self.register_buffer('pe', pe)  # max_length*1*hid_dim

    def forward(self, position_ids):
        
        """
        x: 输入张量，
        """
        position_embeddings = self.pe[position_ids, :]  # 根据位置索引选择位置编码
        return position_embeddings

# -----------------------------4.3.1 attention-Enhanced Transformer Encoder----------------------------------
class Encoder(nn.Module):
    def __init__(self, parameters,n_head=1, n_layer=1):
        super().__init__()
        self.device = parameters.device
        self.hid_dim = parameters.hid_dim
        self.hidden_dim = parameters.hidden_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_features_flag = parameters.online_features_flag
        self.pro_features_flag = parameters.pro_features_flag
        self.n_head = n_head
        self.n_layer = n_layer
        self.params = parameters
        self.pos_dim = parameters.pos_dim
        self.Add_Traj_Representation_flag = parameters.Add_Traj_Representation_flag
        
        # 1  basis feature

        input_dim = 3 # src.shape[2]
        if self.Add_Traj_Representation_flag:

            input_dim+=1+1+parameters.online_dim # speed,src_rid,online_dim
 
            # 2 trajectory features
            self.embedding_T = TimeEmbedding(parameters,self.device)
            
            input_dim+=parameters.time_dim
            
            self.grid_dict = get_grid_dict(parameters.max_xid,parameters.max_yid)
            self.grid_emb = nn.Embedding(len(self.grid_dict.values())+1,parameters.grid_dim)
            input_dim += parameters.grid_dim
        
            self.pos_encoder = PositionalEncoding(self.pos_dim)
            input_dim+=parameters.pos_dim
            
        else:
            input_dim+=1+parameters.online_dim # speed,online_dim

        # 3  graph network

        self.Add_Graph_Representation_flag = parameters.Add_Graph_Representation_flag
        self.Get_Graph_Emb =Get_Graph_Emb(parameters)

        if self.Add_Graph_Representation_flag:
            self.RoadGNN= RoadGNN(parameters)
            self.road_network = parameters.road_network.to(self.device)
            input_dim +=parameters.hid_dim 
            self.ProbGAT = ProbGAT(parameters)
            input_dim+=parameters.hid_dim


        self.co_attention = Co_Att(self.hid_dim).to(self.device)

        self.Add_transformer_ST_flag = parameters.Add_transformer_ST_flag

        if self.Add_transformer_ST_flag:  
            self.linear = nn.Linear(input_dim, self.hid_dim)
            self.transformer = Transformer(self.hid_dim, self.n_layer,
                                        self.device,parameters)  # GPSTransfomer 包括 Transformer,GRL,GREADOUT
        else:
            self.rnn = nn.GRU(input_dim, self.hid_dim)
        
        self.dropout = nn.Dropout(parameters.dropout)

        if self.pro_features_flag:
            self.extra = Extra_MLP(parameters)
            self.fc_hid = nn.Linear(self.hid_dim + self.pro_output_dim, self.hid_dim)
    def normalize(self, x):
    # 实现数据规范化逻辑，根据数据的特点进行相应的处理
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        # 处理第一维全为0的情况
        mask = (std == 0.0)
        std = torch.where(mask, torch.ones_like(std), std)
        normalized_x = torch.where(mask, x, (x - mean) / std)
        return normalized_x
    def forward(self,choose_norm,src, src_len, pro_features,src_rids,src_timeids,src_road_feas,src_point_feas,src_gps_seqs):
        
        # road_emb  id_size*embedding_size
        # src = [src len, batch size, 3]                                                                   
        # src_len = [batch size]
        # pro_features 1 uid +24 hour + holiday
        # src_rids seq_len*batch_size*1
        # src_road_feas seq_len*batch_size*8  # 'lat_begin','lat_end','lng_begin','lng_end','code','norm_num_start_neighbors','norm_num_end_neighbors','norm_len'
        # src_point_feas (speed seq_len*batch_size*1 )
        # src_gps_seqs (location seq_batch_size*2 )
        src_raw_id = src[:, :, 2].unsqueeze(2)  # 获取位置索引
        if self.Add_Traj_Representation_flag:
            # 1 basis feature
            src_point_feas = self.normalize(src_point_feas)
            src_emb_1 = torch.cat((src,src_rids,src_point_feas,src_road_feas),dim = -1)
            
            # 2 trajectory feature

            t_emb = self.embedding_T(src_timeids.permute(1,0,2),src_len).permute(1,0,2) # (B, T, d_model)

            grid = src[:,:,:2].to(torch.int64) # seq_len*bath_size*2
            grid = torch.tensor([[[self.grid_dict[(col[0].cpu().tolist(),col[1].cpu().tolist())]] for col in row ] for row in src ]).to(torch.int64).to(self.device)
            grid = grid.squeeze()
            grid_emb  = self.grid_emb(grid)

            src_id = src[:, :, 2].long().squeeze()  # 获取每个序列位置的位置索引
            src_id_emb=  self.pos_encoder(src_id)

            src_emb_2 = torch.cat((t_emb,grid_emb,src_id_emb),dim = -1)

            src = torch.cat((src_emb_1, src_emb_2), dim=-1)
        else:
            src = torch.cat((src,src_point_feas,src_road_feas), dim=-1)
        
        if self.Add_Graph_Representation_flag:
            node_embeddings_vec = self.RoadGNN(self.road_network) # [id size, embedding_size]
            att_emb_1= self.Get_Graph_Emb(node_embeddings_vec,src_rids) # seq_len*batch_size_embedding_size #
        
            node_embeddings_prob = self.ProbGAT(choose_norm)
            att_emb_2 = self.Get_Graph_Emb(node_embeddings_prob,src_rids) # seq_len*batch_size_embedding_size 
            
            att_emb_1, att_emb_2 = self.co_attention(att_emb_1,att_emb_2)
            att_emb = torch.cat((att_emb_1,att_emb_2),dim = -1)
            src_emb_3 = att_emb.permute(1,0,2)
            src = torch.cat((src,src_emb_3), dim=-1)
        else:
            pass

        if self.Add_transformer_ST_flag:
            max_src_len = src.size(0)
            bs = src.size(1)
            mask3d = torch.zeros(bs, max_src_len, max_src_len).to(self.device)
            for i in range(bs):
                mask3d[i, :src_len[i], :src_len[i]] = 1
            src = self.linear(src)
            src = src.transpose(0, 1)
        
            temporal_mat = cal_temporal_mat(src_raw_id.permute(1,0,2)).to(self.device)
            dis_mat = cal_dis_mat(src_gps_seqs.permute(1,0,2)).to(self.device)

            outputs = self.transformer(src,temporal_mat,dis_mat,mask3d)  # transformer 
            outputs = outputs.transpose(0, 1)  # [src len, bs, hid dim]
        
            for i in range(bs):
                outputs[src_len[i]:, i, :] = 0
            hidden = torch.mean(outputs, dim=0).unsqueeze(0)
        else:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)
            packed_outputs, hidden = self.rnn(packed_embedded)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        #  the input to the transformer should be  (S, N, E)
        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer

        # hidden = [1, batch size, hidden_dim]
        # outputs = [src len, batch size, hidden_dim * num directions]
            
        if self.pro_features_flag:
            extra_emb = self.extra(pro_features)
            extra_emb = extra_emb.unsqueeze(0) # extra_emb = [1, batch size, extra output dim]
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=2)))
            # hidden = [1, batch size, hidden dim]
     
        # outputs = [src len, batch size, hidden_dim * num directions]
        return outputs, hidden

#---------------------------------- 4.3.2  Feature Differences-Aware Decoder---------------------------
class Attention(nn.Module):
    # TODO update to more advanced attention layer.
    def __init__(self, parameters):
        super().__init__()
        self.params = parameters
        self.hid_dim = parameters.hid_dim
        self.Add_feature_differences_flag = parameters.Add_feature_differences_flag
        
        if self.Add_feature_differences_flag:
            self.attn = nn.Linear(self.hid_dim * 2+3, self.hid_dim)
        else:
            self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)
        self.device = parameters.device

    def forward(self,t,hidden, encoder_outputs,attn_mask,src_gps_seqs,src,src_rids,input_id,trg_gps_seqs):
        # hidden = [1, bath size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        # repeat decoder hidden sate src_len times
        hidden = hidden.repeat(src_len, 1, 1) # src_len*batch_size*hid_dim
        hidden = hidden.permute(1, 0, 2) # batch_size*src_len*hid_dim
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [ batch size，src_len, hid dim * num directions]
        # hidden = [batch size, src len, hid dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]

        src_gps_seqs = src_gps_seqs.permute(1,0,2) # batch_size*src_len*2
        src = src.permute(1,0,2) # batch_size*src_len*3
        src_rids = src_rids.permute(1,0,2)  # batch_size*src_len*3
        input_id = input_id.view(batch_size,-1)
        # trg_gps_seqs batch_size*2
        if self.Add_feature_differences_flag:
            rel_matrix = get_gps_distance(src_gps_seqs,trg_gps_seqs,src_len).to(self.device)
            distance_id = get_ID_distance(t,src,src_len).to(self.device)
            prob_minus = get_road_prob_minus(src_rids,input_id,src_len,self.road_emb).to(self.device)
            energy = torch.tanh(self.attn(torch.cat((hidden, rel_matrix,distance_id,prob_minus,encoder_outputs), dim=2)))
        else:
            energy = torch.tanh(self.attn(torch.cat((hidden,encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]
        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        # using mask to force the attention to only be over non-padding elements.

        return F.softmax(attention, dim=1)
    
# s𝑖 = 𝐺𝑅𝑈 (s𝑖−1, g𝑖−1, fs 𝑗−1),h𝑗 = 𝐺𝑅𝑈 (h𝑗−1, 𝑒𝑗−1, 𝑟 𝑗−1, a𝑗 , fs 𝑗−1).
# use road_emb and change the attention count method
# decoder
class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.params = parameters
        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag

        rnn_input_dim = self.id_emb_dim + 1
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim
        type_input_dim = self.id_emb_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
                          nn.Linear(type_input_dim, self.hid_dim),
                          nn.ReLU()
                          )
        
   
        if self.attn_flag: # a
            self.attn = Attention(parameters)
            rnn_input_dim = rnn_input_dim + self.hid_dim 

        if self.online_features_flag: #f
            rnn_input_dim = rnn_input_dim + self.online_dim  # 5 poi and 5 road network
            
        if self.tandem_fea_flag:  # r
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim

        self.emb_id = nn.Embedding(parameters.id_size, parameters.id_emb_dim)

        self.rnn = nn.GRU(rnn_input_dim, self.hid_dim)
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)
        self.device = parameters.device
        
        
    def forward(self,t,input_id, input_rate, hidden, encoder_outputs, attn_mask, constraint_vec, pro_features, online_features, rid_features,src_gps_seqs,src,src_rids,input_tra_gps):

        # input_id = [batch size, 1] rid long
        # input_rate = [batch size, 1] rate float. 
        # hidden = [1, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        # attn_mask = [batch size, src len]
        # pre_grid = [batch size, 3]
        # next_grid = [batch size, 3]
        # constraint_vec = [batch size, id_size], [id_size] is the vector of reachable rid
        # pro_features = [batch size, profile features input dim]
        # online_features = [batch size, online features dim]
        # rid_features = [batch size, rid features dim]
        
        input_id = input_id.squeeze(1).unsqueeze(0)  # cannot use squeeze() bug for batch size = 1
        # input_id = [1, batch size]
        input_rate = input_rate.unsqueeze(0)
        # input_rate = [1, batch size, 1]
        #  enbedded  𝑒𝑗−1
       
        embedded = self.dropout(self.emb_id(input_id))

        
        # embedded = [1, batch size, emb dim]

        if self.attn_flag:
            a = self.attn(t,hidden, encoder_outputs, attn_mask,src_gps_seqs,src,src_rids,input_id,input_tra_gps)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs = [batch size, src len, hid dim * num directions]
            weighted = torch.bmm(a, encoder_outputs)
            # weighted = [batch size, 1, hid dim * num directions]
            weighted = weighted.permute(1, 0, 2)
            # weighted = [1, batch size, hid dim * num directions]

            if self.online_features_flag:
                
                rnn_input = torch.cat((weighted, embedded, input_rate, 
                                       online_features.unsqueeze(0)), dim=2) # 𝑒𝑗−1, 𝑟 𝑗−1, a𝑗 , fs 𝑗−1
            else:
               
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)  
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((embedded, input_rate), dim=2)

        output, hidden = self.rnn(rnn_input, hidden) #hidden (h𝑗−1).
        
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # assert (output == hidden).all()

        # pre_rid
        if self.dis_prob_mask_flag:
            prediction_id = mask_log_softmax(self.fc_id_out(output.squeeze(0)), 
                                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(self.fc_id_out(output.squeeze(0)), dim=1)
            # then the loss function should change to nll_loss()

        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
       
        id_emb = self.dropout(self.emb_id(max_id))
        rate_input = torch.cat((id_emb, hidden.squeeze(0)),dim=1)
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, online_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))
        

        # prediction_id = [batch size, id_size]
        # prediction_rate = [batch size, 1]

        return prediction_id, prediction_rate, hidden


# ------------------------4.3 Spatiotemporal Interval-Informed Seq2Seq -------------------------------------
class Seq2SeqMulti(nn.Module):
    def __init__(self, encoder, decoder, device,parameters):
        super().__init__()
    
        self.device = device
        self.encoder = encoder  # Encoder
        self.decoder = decoder  # DecoderMulti
        self.params = parameters

    def forward(self,choose_norm,src,src_gps_seqs, pro_features,src_point_feas,src_rids,src_timeids,src_road_feas,src_len, trg_id, trg_rate,trg_gps_seqs, trg_len\
                                         ,constraint_mat,online_features_dict,rn_dict,rid_features_dict,teacher_forcing_ratio=0.5):

        """
        src = [src len, batch size, 3], x,y,t
        src_len = [batch size]
        trg_id = [trg len, batch size, 1]
        trg_rate = [trg len, batch size, 1]
        trg_len = [batch size]
        pre_grids = [trg len, batch size, 3]
        nex_grids = [trg len, batch size, 3]
        constraint_mat = [trg len, batch size, id_size]
        pro_features = [batch size, profile features input dim]
        online_features_dict = {rid: online_features} # rid --> grid --> online features
        rid_features_dict = {rid: rn_features}
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        Return:
        ------
        outputs_id: [seq len, batch size, id_size(1)] based on beam search
        outputs_rate: [seq len, batch size, 1]
        """
        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        # road_emb = self.gnn(F.leaky_relu(self.road_emb_id)) # [id size, hidden dim]

        encoder_outputs, hiddens = self.encoder(choose_norm,src, src_len, pro_features, src_rids,src_timeids,src_road_feas,src_point_feas,src_gps_seqs)
        self.decoder.attn.road_emb = choose_norm

        if self.decoder.attn_flag:
            attn_mask = torch.zeros(batch_size, max(src_len))  # only attend on unpadded sequence
            for i in range(len(src_len)):
                attn_mask[i][:src_len[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None
        
        outputs_id, outputs_rate = self.normal_step(max_trg_len, batch_size, trg_id, trg_rate, trg_len,
                                                    encoder_outputs, hiddens, attn_mask,
                                                    online_features_dict,
                                                    rid_features_dict, constraint_mat, pro_features,
                                                    src_gps_seqs,src,src_rids,trg_gps_seqs,rn_dict,
                                                    teacher_forcing_ratio)
   

        return outputs_id, outputs_rate

    def normal_step(self,max_trg_len, batch_size, trg_id, trg_rate, trg_len, encoder_outputs, hidden,
                    attn_mask, online_features_dict, rid_features_dict, constraint_mat, pro_features,src_gps_seqs,src,src_rids,trg_gps_seqs,rn_dict,teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)

        # first input to the decoder is the <sos> tokens
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        input_tra_gps = trg_gps_seqs[0]

        for t in range(1, max_trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and attn_mask
            # receive output tensor (predictions) and new hidden state
            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros((1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None
            
            prediction_id, prediction_rate, hidden = self.decoder(t,input_id, input_rate, hidden, encoder_outputs,
                                                                     attn_mask,\
                                                                     constraint_mat[t], pro_features, online_features,
                                                                     rid_features,src_gps_seqs,src,src_rids,input_tra_gps)

            # place predictions in a tensor holding predictions for each token
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1_id = prediction_id.argmax(1)
            top1_id = top1_id.unsqueeze(-1)  # make sure the output has the same dimension as input
            
            trg_pre_gps = [[rate2gps(rn_dict, s_id[0], s_rate[0], self.params).lat,rate2gps(rn_dict, s_id[0], s_rate[0], self.params).lng] for idx, (s_id, s_rate) in enumerate(zip(top1_id, prediction_rate))]
            trg_pre_gps =  torch.tensor(trg_pre_gps).to(self.device)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate
            input_tra_gps = trg_gps_seqs[t] if teacher_force else trg_pre_gps
        # max_trg_len, batch_size, trg_rid_size
        outputs_id = outputs_id.permute(1, 0, 2)  # batch size, seq len, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1
        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = 0
            outputs_id[i][trg_len[i]:, 0] = 1  # make sure argmax will return eid0
            outputs_rate[i][trg_len[i]:] = 0
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate