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
from collections import Counter
from utils.utils import save_json_data, create_dir, load_pkl_data
from pre_processing.mbr import MBR
from pre_processing.road_network import load_rn_shp,load_GNN_graph
from pre_processing.road_network_route import load_rn_shp_route
from pre_processing.road_network_route_osm import load_rn_shp_route_osm
from pre_processing.get_node2vec import load_GNN_netowrk

from pre_processing.datasets import Dataset, collate_fn, split_data
from models.loss_fn import cal_id_acc, check_rn_dis_loss
from models.model import Encoder, DecoderMulti, Seq2SeqMulti
from multi_train import  init_weights,train,evaluate,test,get_test_time
from models.model_utils import get_rid_grid,toseq, get_constraint_mask,load_rid_freqs,rate2gps
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict,get_rid_grid_dict

parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
parser.add_argument('--city', type=str, default='Porto')
parser.add_argument('--epochs', type=int, default=1, help='epochs')
parser.add_argument('--i', type=int, default=0, help='GPU index')
parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
parser.add_argument('--ds_type', type=str, default='random', help='sample type')
parser.add_argument('--module_type', type=str, default='simple', help='module type')
parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task rate')
parser.add_argument('--hid_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
parser.add_argument('--test_code', action='store_false', help='test the code or not')
parser.add_argument('--dis_prob_mask_flag', action='store_false', help='flag of using prob mask')
parser.add_argument('--pro_features_flag', action='store_false', help='flag of using profile features')
parser.add_argument('--online_features_flag', action='store_false', help='flag of using online features')
parser.add_argument('--tandem_fea_flag', action='store_false', help='flag of using tandem rid features')
parser.add_argument('--no_attn_flag', action='store_false', help='flag of using attention')
parser.add_argument('--load_pretrained_flag', action='store_true', help='flag of load pretrained model')
parser.add_argument('--model_old_path', type=str, default='', help='old model path')
parser.add_argument('--no_debug', action='store_false', help='flag of debug')
parser.add_argument('--train_flag', action='store_false', help='flag of training')
parser.add_argument('--test_flag', action='store_false', help='flag of testing')

parser.add_argument('--Add_Graph_Representation_flag', action='store_false', help='Add_Graph_Representation')
parser.add_argument('--Add_change_GNN_flag', action='store_true', help='change_candiGNN_to_GCN')

parser.add_argument('--Add_Traj_Representation_flag', action='store_false', help='Add_Traj_Representation')

parser.add_argument('--Add_transformer_ST_flag', action='store_false', help='Add_transformer_ST')
parser.add_argument('--Add_feature_differences_flag', action='store_false', help='Add_feature_differences_Decoder')

opts  = parser.parse_known_args()[0]
debug = opts.no_debug
device = torch.device('cuda:{}'.format(opts.i) if torch.cuda.is_available() else 'cpu')
print(device)


save_path = './Trajectory_Reconstruction/data_experiment'
E_data_name = opts.city
save_path_map = '{}/{}'.format(save_path,E_data_name)

if opts.city == "Porto":
    zone_range = [41.121621, -8.644531, 41.167815, -8.596830]
    id_size = 5121
    time_span = 15 # sampling rate
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
    'test_code':opts.test_code,

        # pre train
    'load_pretrained_flag':opts.load_pretrained_flag,
    'model_old_path':opts.model_old_path,
    'train_flag':opts.train_flag,
    'test_flag':opts.test_flag,

    # attention
    'attn_flag':opts.no_attn_flag,

    # constranit
    'dis_prob_mask_flag':opts.dis_prob_mask_flag,
    'search_dist':50,
    'beta':15,

    # features
    'tandem_fea_flag':opts.tandem_fea_flag,
    'pro_features_flag':opts.pro_features_flag,
    'online_features_flag':opts.online_features_flag,

    # extra info module 
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
    
    'time_dim':8,
    'grid_dim':32,
    'pos_dim':8,
    
    # GNN parms
    "hidden_dim":64,
    "embedding_size":128,
    'u_norm_dim':8,
    'id_norm_dim':64,

    'lambda1':opts.lambda1,
    'n_epochs':opts.epochs,
    'batch_size':128,
    'learning_rate':1e-3,
    'tf_ratio':0.5,
    'clip':1,
    'log_step':1,

    # random seed
    'seed_train':12345,
    'seed_valid':67890,
    'seed_test':24680,
    'Add_Graph_Representation_flag':opts.Add_Graph_Representation_flag, 
    'Add_change_GNN_flag':opts.Add_change_GNN_flag,
    'Add_Traj_Representation_flag':opts.Add_Traj_Representation_flag,

    'Add_transformer_ST_flag':opts.Add_transformer_ST_flag, 
    'Add_feature_differences_flag':opts.Add_feature_differences_flag, 
}
args.update(args_dict)

if args.load_pretrained_flag:
        model_save_path = args.model_old_path
else:
     ablation_name = ''
     if  args.Add_Graph_Representation_flag == True:
         ablation_name = ablation_name +'_'+ "Add_Graph_Representation_flag"
     if  args.Add_change_GNN_flag == True:
         ablation_name = ablation_name +'_'+ "Add_change_GNN_flag"
     if  args.Add_Traj_Representation_flag == True:
         ablation_name = ablation_name  +'_'+ "Add_Traj_Representation_flag"
         
     if  args.Add_transformer_ST_flag == True:
         ablation_name = ablation_name +'_'+ "Add_transformer_ST_flag"
     if  args.Add_feature_differences_flag == True:
         ablation_name = ablation_name +'_'+ "Add_feature_differences_flag"

     model_save_path = './results/'+'{}/'.format(opts.city)+'keep-ratio_'+str(args.keep_ratio)+'_ds_type_'+str(args.ds_type)+'/'\
     +time.strftime("%Y%m%d_%H%M%S") + ablation_name +'/'
     create_dir(model_save_path)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=model_save_path + 'log.txt',
                    filemode='a')

new2raw_rid_dict = load_rid_freqs(save_path_map, file_name='/new2raw_rid.json') # len(raw2new_rid_dict) 2571
raw2new_rid_dict = load_rid_freqs(save_path_map, file_name='/raw2new_rid.json')
rn_dict = {raw2new_rid_dict[key]:value for (key,value) in raw_rn_dict.items() if key in raw2new_rid_dict.keys()}
mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args.grid_size, rn_dict)
road_network = load_GNN_netowrk(edge_index,'{}'.format(save_path_map))

#online_features_dict 是对rn_dict 的扩充和细化,rn_dict 可以直接索引坐标信息
norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None
rid_features_dict = get_rid_rnfea_dict(raw_rn_dict, args)
rid_features_dict = {raw2new_rid_dict[key]:value for (key,value) in rid_features_dict.items() if key in raw2new_rid_dict.keys()}
online_features_dict = rid_features_dict
args_dict['max_xid'] = max_xid
args_dict['max_yid'] = max_yid
args_dict['grid_num'] = (max_xid +1, max_yid+1)
args_dict['online_dim'] = len(rid_features_dict[0]) - 4 
args_dict['rid_fea_dim'] = args_dict['online_dim'] 
args_dict['way_type_number'] = len(rid_features_dict[0]) - 7
args.update(args_dict)
logging.info(args_dict)
#  here  we can change the contents in args
args.new2raw_rid_dict = new2raw_rid_dict
args.raw2new_rid_dict = raw2new_rid_dict
args.neighbor_all = neighbor_all
args.road_network = road_network
args.rn = rn
# online_dim : 'lat_begin','lat_end','lng_begin','lng_end',len(code):7,'norm_num_start_neighbors','norm_num_end_neighbors','norm_len'

train_trajs_dir = "{}/model_data/train_data/train.npy".format(save_path_map)
valid_trajs_dir =  "{}/model_data/valid_data/val.npy".format(save_path_map)
test_trajs_dir = "{}/model_data/test_data/test.npy".format(save_path_map)
if args.test_code:
    test_dataset = Dataset(test_trajs_dir, mbr,online_features_dict,raw2new_rid_dict,\
                        parameters=args,seed_value = args.seed_test)
    
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,\
                                            shuffle=args.shuffle, collate_fn=lambda data: collate_fn(data, rn, args),\
                                            num_workers=4, pin_memory=True)
    train_iterator = test_iterator
    valid_iterator = test_iterator
else:


    train_dataset = Dataset(train_trajs_dir, mbr,online_features_dict,raw2new_rid_dict,\
                            parameters=args,seed_value = args.seed_train)
    valid_dataset = Dataset(valid_trajs_dir, mbr,online_features_dict,raw2new_rid_dict,\
                            parameters=args,seed_value = args.seed_valid)
    test_dataset = Dataset(test_trajs_dir, mbr,online_features_dict,raw2new_rid_dict,\
                            parameters=args,seed_value = args.seed_test)

    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    print('test dataset shape: ' + str(len(test_dataset)))


    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    shuffle=args.shuffle, collate_fn=lambda data: collate_fn(data, rn, args),
                                                num_workers=4, pin_memory=True)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                    shuffle=args.shuffle, collate_fn=lambda data: collate_fn(data, rn, args),
                                                num_workers=4, pin_memory=True)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,\
                                                shuffle=args.shuffle, collate_fn=lambda data: collate_fn(data, rn, args),\
                                                num_workers=4, pin_memory=True)

    logging.info('Finish data preparing.')
    logging.info('training dataset shape: ' + str(len(train_dataset)))
    logging.info('validation dataset shape: ' + str(len(valid_dataset)))
    logging.info('test dataset shape: ' + str(len(test_dataset)))

print('training dataloader shape: ' + str(len(train_iterator)))
print('validation dataloader shape: ' + str(len(valid_iterator)))
print('test dataset dataloader: ' + str(len(test_iterator)))

args.Add_Graph_Representation_flag = False
args.Add_change_GNN_flag =False
args.Add_Traj_Representation_flag =True
    
args.Add_transformer_ST_flag =True
args.Add_feature_differences_flag =True


enc = Encoder(args)
dec = DecoderMulti(args)
model = Seq2SeqMulti(enc, dec, device,args).to(device)
model.apply(init_weights)  # learn how to init weights
if args.load_pretrained_flag:
    model.load_state_dict(torch.load(args.model_old_path + 'val-best-model.pt', map_location=device))
print('model', str(model))
logging.info('model' + str(model))

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model params num : {}'.format(num_params))
logging.info('model_param_number : ' + str(num_params))

if args.train_flag:
    ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision,ls_train_id_f1, \
    ls_train_rate_loss, ls_train_id_loss = [], [], [], [], [], [],[]
    ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision,ls_valid_id_f1, \
    ls_valid_dis_mae_loss, ls_valid_dis_rmse_loss = [], [], [], [], [], [],[]
    ls_valid_dis_rn_mae_loss, ls_valid_dis_rn_rmse_loss, ls_valid_rate_loss, ls_valid_id_loss = [], [], [], []

    dict_train_loss = {}
    dict_valid_loss = {}
    best_valid_loss = float('inf')  # compare id loss
    best_acc = 0

    # get all parameters (model parameters + task dependent log variances)
    log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in tqdm(range(args.n_epochs)):
        start_time = time.time()

        new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision,train_id_f1, \
        train_rate_loss, train_id_loss = train(model, train_iterator, optimizer, log_vars,rn_dict,\
                                                online_features_dict, rid_features_dict, args)

        valid_id_acc1, valid_id_recall, valid_id_precision,valid_id_f1, valid_dis_mae_loss, valid_dis_rmse_loss, \
        valid_rate_loss, valid_id_loss = evaluate(model, valid_iterator,\
                                                    rn_dict,rn_route,\
                                                    online_features_dict, rid_features_dict, raw_rn_dict,
                                                    new2raw_rid_dict, args)
    
        ls_train_loss.append(train_loss)
        ls_train_id_acc1.append(train_id_acc1)
        ls_train_id_recall.append(train_id_recall)
        ls_train_id_precision.append(train_id_precision)
        ls_train_id_f1.append(train_id_f1)
        ls_train_rate_loss.append(train_rate_loss)
        ls_train_id_loss.append(train_id_loss)

        ls_valid_id_acc1.append(valid_id_acc1)
        ls_valid_id_recall.append(valid_id_recall)
        ls_valid_id_precision.append(valid_id_precision)
        ls_valid_id_f1.append(valid_id_f1)
        ls_valid_dis_mae_loss.append(valid_dis_mae_loss)
        ls_valid_dis_rmse_loss.append(valid_dis_rmse_loss)
        ls_valid_rate_loss.append(valid_rate_loss)
        ls_valid_id_loss.append(valid_id_loss)
        valid_loss = valid_rate_loss + valid_id_loss
        ls_valid_loss.append(valid_loss)

        dict_train_loss['train_ttl_loss'] = ls_train_loss
        dict_train_loss['train_id_acc1'] = ls_train_id_acc1
        dict_train_loss['train_id_recall'] = ls_train_id_recall
        dict_train_loss['train_id_precision'] = ls_train_id_precision
        dict_train_loss['train_id_f1'] = ls_train_id_f1
        dict_train_loss['train_rate_loss'] = ls_train_rate_loss
        dict_train_loss['train_id_loss'] = ls_train_id_loss

        dict_valid_loss['valid_ttl_loss'] = ls_valid_loss
        dict_valid_loss['valid_id_acc1'] = ls_valid_id_acc1
        dict_valid_loss['valid_id_recall'] = ls_valid_id_recall
        dict_valid_loss['valid_id_precision'] = ls_valid_id_precision
        dict_valid_loss['valid_id_f1'] = ls_valid_id_f1
        dict_valid_loss['valid_rate_loss'] = ls_valid_rate_loss
        dict_valid_loss['valid_dis_mae_loss'] = ls_valid_dis_mae_loss
        dict_valid_loss['valid_dis_rmse_loss'] = ls_valid_dis_rmse_loss
        dict_valid_loss['valid_id_loss'] = ls_valid_id_loss

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path + 'val-best-model.pt')
     
            logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
            weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
            logging.info('log_vars:' + str(weights))
            logging.info('\tTrain Loss:' + str(train_loss) +
                            '\tTrain RID Acc1:' + str(train_id_acc1) +
                            '\tTrain RID Recall:' + str(train_id_recall) +
                            '\tTrain RID Precision:' + str(train_id_precision) +
                            '\tTrain RID f1:' + str(train_id_f1) +
                            '\tTrain Rate Loss:' + str(train_rate_loss) +
                            '\tTrain RID Loss:' + str(train_id_loss))
            logging.info('\tValid Loss:' + str(valid_loss) +
                            '\tValid RID Acc1:' + str(valid_id_acc1) +
                            '\tValid RID Recall:' + str(valid_id_recall) +
                            '\tValid RID Precision:' + str(valid_id_precision) +
                            '\tValid RID f1:' + str(valid_id_f1) +
                            '\tValid Distance MAE Loss:' + str(valid_dis_mae_loss) +
                            '\tValid Distance RMSE Loss:' + str(valid_dis_rmse_loss) +
                            '\tValid Rate Loss:' + str(valid_rate_loss) +
                            '\tValid RID Loss:' + str(valid_id_loss))

            # torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
            save_json_data(dict_train_loss, model_save_path, "train_loss.json")
            save_json_data(dict_valid_loss, model_save_path, "valid_loss.json")

if args.test_flag:
    model.load_state_dict(torch.load(model_save_path + 'val-best-model.pt', map_location=device))
    start_time = time.time()
    test_id_acc1, test_id_recall, test_id_precision,test_id_f1, test_dis_mae_loss, test_dis_rmse_loss, \
    test_dis_rn_mae_loss, test_dis_rn_rmse_loss, test_rate_loss, test_id_loss = test(model, test_iterator,
                                                                                            rn_dict, rn_route,
                                                                                            online_features_dict,
                                                                                            rid_features_dict,
                                                                                            raw_rn_dict, new2raw_rid_dict,
                                                                                            args)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    logging.info('Test Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
    logging.info('\tTest RID Acc1:' + str(test_id_acc1) +
                    '\tTest RID Recall:' + str(test_id_recall) +
                    '\tTest RID Precision:' + str(test_id_precision) +
                      '\tTest RID f1:' + str(test_id_f1) +
                    '\tTest Distance MAE Loss:' + str(test_dis_mae_loss) +
                    '\tTest Distance RMSE Loss:' + str(test_dis_rmse_loss) +
                    '\tTest Distance RN MAE Loss:' + str(test_dis_rn_mae_loss) +
                    '\tTest Distance RN RMSE Loss:' + str(test_dis_rn_rmse_loss) +
                    '\tTest Rate Loss:' + str(test_rate_loss) +
                    '\tTest RID Loss:' + str(test_id_loss))
    recovery_time,memory_used = get_test_time(model, test_iterator,\
                                                                                            rn_dict, rn_route,\
                                                                                            online_features_dict,\
                                                                                            rid_features_dict,\
                                                                                            raw_rn_dict, new2raw_rid_dict,\
                                                                                            args)
    recovery_time_single = recovery_time/len(test_dataset)
    memory_used_single = memory_used/len(test_dataset)
    logging.info('\tALL_recovery_time :' + str(recovery_time) + 's'+
             '\tlen(test_dataset):' + str(len(test_dataset))  +
                '\trecovery_time_single:' + str(recovery_time_single)+ 's'+
               '\tmemory_used_single:' +str(memory_used_single)+'MB')


