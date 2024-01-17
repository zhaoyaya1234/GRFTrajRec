import time
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import random
from collections import Counter
from utils.utils import save_json_data, create_dir, load_pkl_data
from pre_processing.mbr import MBR
from pre_processing.road_network import load_rn_shp,load_GNN_graph
from pre_processing.road_network_route import load_rn_shp_route
from pre_processing.road_network_route_osm import load_rn_shp_route_osm
from pre_processing.get_node2vec import load_netowrk
from pre_processing.map import get_candi,get_candi_proba
from pre_processing.datasets import Dataset, collate_fn, split_data
from models.loss_fn import cal_id_acc, check_rn_dis_loss
from models.model import Encoder, DecoderMulti, Seq2SeqMulti
from multi_train import  init_weights,train,evaluate,test
from models.model_utils import get_rid_grid,toseq, get_constraint_mask,load_rid_freqs,rate2gps
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict,get_rid_grid_dict
from utils.utils import get_min_area,get_conti_df,get_nonan_df

parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
parser.add_argument('--city', type=str, default='Porto')
parser.add_argument('--i', type=str, default='0', help='the data num')
parser.add_argument('--ds_type', type=str, default='random', help='sample type')
parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task rate')
parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
parser.add_argument('--epochs', type=int, default=1, help='epochs')
parser.add_argument('--module_type', type=str, default='simple', help='module type')
parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
parser.add_argument('--dis_prob_mask_flag', action='store_true', help='flag of using prob mask')
parser.add_argument('--pro_features_flag', action='store_true', help='flag of using profile features')
parser.add_argument('--online_features_flag', action='store_true', help='flag of using online features')
parser.add_argument('--tandem_fea_flag', action='store_true', help='flag of using tandem rid features')
parser.add_argument('--no_attn_flag', action='store_false', help='flag of using attention')
parser.add_argument('--load_pretrained_flag', action='store_true', help='flag of load pretrained model')
parser.add_argument('--model_old_path', type=str, default='', help='old model path')
parser.add_argument('--no_debug', action='store_false', help='flag of debug')
parser.add_argument('--no_train_flag', action='store_false', help='flag of training')
parser.add_argument('--test_flag', action='store_true', help='flag of testing')

opts  = parser.parse_known_args()[0]
debug = opts.no_debug
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
print(opts.city)


save_path = './Trajectory_Reconstruction/data_experiment'
E_data_name = opts.city
save_path_map = '{}/{}'.format(save_path,E_data_name)
save_path_rn = "{}/RN_model_data".format(save_path_map)

if opts.city == "Porto":
    zone_range = [41.121621, -8.644531, 41.167815, -8.596830]
    id_size = 5121
    time_span = 15
if opts.city == "yancheng":
    zone_range = [33.1696824,120.1070088,33.4401188,120.3560447]
    id_size = 7061
    time_span = 15 # sampling rate
    
print('-----------------------1 load raw graph----------------------')

if opts.city == 'Porto':
    rn_dir = '{}/{}_route_graph.gpickle'.format(save_path_map,E_data_name)
    rn_route = load_rn_shp_route(rn_dir, is_directed=True)
elif opts.city == 'yancheng':
    rn_dir =  '{}/{}osm1'.format(save_path_map,E_data_name)
    rn_route = load_rn_shp_route_osm(rn_dir, is_directed=True)

print('---------------------2 load self generated graph---------------')
rn_graph_dir = '{}/{}_graph.gpickle'.format(save_path_map,E_data_name)
rn,raw_rn_dict= load_rn_shp(rn_graph_dir, is_directed=True)# 加载路网数据
print('---------------------3 load GNN graph---------------')
GNN_graph_dir = '{}/GNN_graph.gpickle'.format(save_path_map,E_data_name)
edge_index,neighbor_all= load_GNN_graph(GNN_graph_dir)

args = AttrDict()
args_dict = {
    'city':opts.city,
    'module_type':opts.module_type,
    'debug':debug,
    'device':device,

        # pre train
    'load_pretrained_flag':opts.load_pretrained_flag,
    'model_old_path':opts.model_old_path,
    'train_flag':True,
    'test_flag':True,

    # attention
    'attn_flag':opts.no_attn_flag,

    # constranit
    'dis_prob_mask_flag':True,
    'search_dist':50,
    'beta':15,

    # features
    'tandem_fea_flag':opts.tandem_fea_flag,
    'pro_features_flag':opts.pro_features_flag,
    'online_features_flag':opts.online_features_flag,

    # extra info module 
    'rid_fea_dim':8,
    'pro_input_dim':26, # 1 [id] +24[hour]+ 1[holiday]
    'pro_output_dim':8,
    'poi_num':5,
    'poi_type':'company,food,shopping,viewpoint,house',
    'uid_num':500,

    # MBR 
    'min_lat':zone_range[0],
    'min_lng':zone_range[1],
    'max_lat':zone_range[2],
    'max_lng':zone_range[3],

    # input data params
    'keep_ratio':opts.keep_ratio,
    'grid_size':opts.grid_size,
    'time_span':time_span, # 连续点间隔1秒
    'win_size':50,
    'ds_type':opts.ds_type,
    'split_flag':True,
    'shuffle':True,
    

    # model params
    'hid_dim':opts.hid_dim,
    'id_emb_dim':128,
    'dropout':0.5,
    'id_size':id_size, 
    'way_type_output':4,


    # GNN parms
    "hidden_dim":64,
    "embedding_size":128,
    'u_norm_dim':8,
    'id_norm_dim':64,

    'lambda1':opts.lambda1,
    'n_epochs':opts.epochs,
    'batch_size':32,
    'learning_rate':1e-3,
    'tf_ratio':0.5,
    'clip':1,
    'log_step':1,
    "online_feature_dim":32,

    # random seed
    'seed_train':12345,
    'seed_valid':67890,
    'seed_test':24680,
    
}
args.update(args_dict)


def inside_zone(zone_range,lat, lon):
    return zone_range[0] <= lat and lat <= zone_range[2] and zone_range[1] <= lon and lon <= \
            zone_range[3]

def is_coordinate_in_zone(coordinate, zone_range):
    latitudes = [value.lat for value in coordinate] 
    longitudes = [value.lng for value in coordinate]
    inzone_flag = True
    for lat, lon in zip(latitudes, longitudes):
        inzone_flag = inzone_flag and inside_zone(zone_range,lat, lon)
    return inzone_flag
valid_edges = []
for key in rn_route.coords_dict.keys():
    if is_coordinate_in_zone(rn_route.coords_dict[key], zone_range):
        valid_edges.append(key)


valid_edges_rn= np.load('{}/valid_edges.npy'.format(save_path_map),allow_pickle=True)
assert (np.array(sorted(valid_edges)) == valid_edges_rn).all()
# # 2  get candi data

i = opts.i
print(i)
data_dir = save_path_map + '/traj_dict_{}.npy'.format(i)
trg_trajs= np.load(data_dir,allow_pickle=True).item()

if opts.city == 'Porto':
    traj_dict_map = get_candi(rn_route,trg_trajs)
if opts.city == 'yancheng':
    traj_dict_map = get_candi_proba(rn_route,trg_trajs)

np.save('{}/traj_dict_map_{}.npy'.format(save_path_map,i),traj_dict_map) # 注意带上后缀名
print('traj_dict_map save well')

data_dir = save_path_map + '/traj_dict_map.npy'
traj_dict= np.load(data_dir,allow_pickle=True).item()
for value in traj_dict.values():
    for i in range(len(value)):
        print(value['error'][i])
    break

# # 3  get traj_dict in valid_edges and user_cent

def merge_traj(num,traj_dict_name):
    traj_dict_merge = {}
    for i in range(num):
        traj_dict_split = np.load('{}/{}_{}.npy'.format(save_path_map,traj_dict_name,i),allow_pickle=True).item()
        for key in tqdm(traj_dict_split.keys()):
            traj_dict_merge[key] = traj_dict_split[key] 
    return traj_dict_merge
if args.city =='Porto':
    traj_dict_merge = merge_traj(11,'traj_dict_map')
if args.city == 'yancheng':
    traj_dict_merge = merge_traj(41,'traj_dict_map')
np.save('{}/traj_dict_map.npy'.format(save_path_map),traj_dict_merge) # 注意带上后缀名

print(len(traj_dict_merge))

# 处理文件里面的NAN
data_dir = save_path_map + '/traj_dict_map.npy'
traj_dict_map= np.load(data_dir,allow_pickle=True).item()
if opts.city == 'yancheng':
    traj_dict_map = get_nonan_df(traj_dict_map)
print('the length of raw traj_dict',len(traj_dict_map))
if opts.city == 'yancheng':
    eid = []
    for value in tqdm(traj_dict_map.values()): 
        for s_eid in value['eid']:
            eid.append(s_eid)
    min_lat,min_lng,max_lat,max_lng = get_min_area(eid,rn_route)
    print(min_lat,min_lng,max_lat,max_lng)
    # 重新改一下各个模型的范围 

if opts.city == 'Porto':
    data_dir = save_path_map + '/traj_dict_map.npy'
    traj_dict= np.load(data_dir,allow_pickle=True).item()
if opts.city == 'yancheng':
    traj_dict = traj_dict_map
traj_dict_last = {}
for key in tqdm(traj_dict.keys()):
    if traj_dict[key]['eid'].isin(valid_edges).all():    
        traj_dict_last[key] = traj_dict[key]
print('the length of  traj_dict_last ',len(traj_dict_last))
# traj_dict_last = {key:traj_dict_last[key] for key in traj_dict_last.keys() if len(traj_dict_last[key])>80}
# uid = []
# for value in traj_dict_last.values():
#     uid.append(value['uid'][0])
# user_cent = np.max(uid)+1
# print('the user_cent of {} is {}'.format(opts.city,user_cent))
# the user_cent of Porto is 442  the length of raw traj_dict_last  107699
# the user_cent of Porto is 419  the length of raw traj_dict_last  21640
# np.save('{}/traj_dict_last.npy'.format(save_path_map),traj_dict_last) # 注意带上后缀名

# 3 split data for RNTrajRec

def data_augment(df_list):
    random_seed = 42  # 你可以选择任何整数作为种子值
    random.seed(random_seed)
    aug_df = []
    for df in df_list:
        for start_row in range(15):
            result_df = df.iloc[start_row::15]
            result_df.reset_index(drop=True, inplace=True) 
            if len(result_df)>10:
                aug_df.append(result_df)
    selected_dfs = sorted(aug_df, key=len, reverse=True)[:len(aug_df)//2]
    keep_count = int(len(selected_dfs)*(2/3))
    selected_dfs = random.sample(selected_dfs, keep_count)
    return selected_dfs

def split_data_for_RN(save_path_map, random_seed=42):
    # 设置随机种子以确保结果可重复
    random.seed(random_seed)

    trg_trajs = np.load('{}/traj_dict_last.npy'.format(save_path_map),allow_pickle=True).item()
    trg_trajs = [value for value in trg_trajs.values()]
    ttl_lens = len(trg_trajs)

    test_inds = random.sample(range(ttl_lens), int(ttl_lens * 0.1))  # 10% 作为测试数据
    tmp_inds = [ind for ind in range(ttl_lens) if ind not in test_inds]
    val_inds = random.sample(tmp_inds, int(ttl_lens * 0.2))  # 20% 作为验证数据
    train_inds = [ind for ind in tmp_inds if ind not in val_inds]  # 70% 作为训练数据

    train_data = [trg_trajs[j] for j in train_inds]
    val_data = [trg_trajs[j] for j in val_inds]
    test_data = [trg_trajs[j] for j in test_inds]
    if opts.city == 'yancheng':
        train_data = data_augment(train_data)
        val_data = data_augment(val_data)
        test_data = data_augment(test_data)
    return train_data, val_data, test_data
def write_txt(output_file,data,data_type):
    with open(output_file, 'w') as file:
        for  df in tqdm(data):
            for index, row in df.iterrows():
                # 提取四列数据并写入txt文件
                timestamp = row['timestamp']
                if data_type == 'input':
                    latitude = row['latitude']
                    longitude = row['longitude']
                else:
                    latitude = row['proj_lat']
                    longitude = row['proj_lng']
                eid = row['eid'] - 1
                file.write(f"{timestamp} {latitude} {longitude} {int(eid)}\n")
            # 在每个DataFrame的末尾写入-1
            file.write("-1\n")
    print('write well')

train_data, val_data, test_data = split_data_for_RN(save_path_map)

output_file = '{}/train/train_input.txt'.format(save_path_rn)
write_txt(output_file,train_data,'input')
output_file = '{}/train/train_output.txt'.format(save_path_rn)
write_txt(output_file,train_data,'output')

output_file = '{}/valid/valid_input.txt'.format(save_path_rn)
write_txt(output_file,val_data,'input')
output_file = '{}/valid/valid_output.txt'.format(save_path_rn)
write_txt(output_file,val_data,'output')

output_file = '{}/test/test_input.txt'.format(save_path_rn)
write_txt(output_file,test_data,'input')
output_file = '{}/test/test_output.txt'.format(save_path_rn)
write_txt(output_file,test_data,'output')

# 4 split data for other model


traj_input_path = '{}/traj_dict_last.npy'.format(save_path_map)
output_dir = "{}/model_data/".format(save_path_map)
split_data(traj_input_path, output_dir,args)
print('split data well')

# # 5 get node2vec embedding and u_norm embedding 

# 这个一张地图行一次就行了 
from pre_processing.get_node2vec import train_epoch,save_embeddings,get_Node2Vec_model
Node2Vec_model = get_Node2Vec_model(id_size,edge_index)
loader = Node2Vec_model.loader(batch_size=128, shuffle=True)
optimizer = torch.optim.SparseAdam(Node2Vec_model.parameters(), lr=0.01)
# Train until delta loss has been reached
train_epoch(Node2Vec_model, loader, optimizer,epoch = 100) # 默认一百
save_embeddings(Node2Vec_model,id_size,save_path_map)  # 把地图的结点EMBEDDING 保存