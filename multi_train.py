import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import toseq, get_constraint_mask
from models.loss_fn import cal_id_acc, check_rn_dis_loss
from tqdm import tqdm
from pre_processing.TimeKeeper import TimeKeeper
# set random seed
SEED = 20202020

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

def print_gradient(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter: {name}, Gradient norm: {torch.norm(param.grad)}")
def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)


def train(model, iterator, optimizer, log_vars, rn_dict, online_features_dict, rid_features_dict, parameters):
    device = parameters.device
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()

    epoch_ttl_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_f1_loss = 0
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    for i, batch in enumerate(iterator):
        src_grid_seqs, src_gps_seqs, src_pro_feas,src_point_feas,src_rids,src_timeids,src_rates,src_road_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths,constraint_mat,choose_norm= batch
        

        constraint_mat = constraint_mat.permute(1, 0, 2).to(device)

        src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
        src_pro_feas = src_pro_feas.float().to(device)
        src_point_feas = src_point_feas.permute(1, 0, 2).to(device)

        src_rids = src_rids.permute(1, 0, 2).long().to(device)
        src_timeids = src_timeids.permute(1, 0, 2).long().to(device)
        src_rates = src_rates.permute(1, 0, 2).to(device)

        src_road_feas = src_road_feas.permute(1, 0, 2).to(device)
        src_gps_seqs = src_gps_seqs.permute(1, 0, 2).to(device)

        trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
        trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
        trg_rates = trg_rates.permute(1, 0, 2).to(device)
        choose_norm= choose_norm.to(device)

        # constraint_mat = [trg len, batch size, id size]
        # src_grid_seqs = [src len, batch size, 2]
        # src_lengths = [batch size]
        # trg_gps_seqs = [trg len, batch size, 2]
        # trg_rids = [trg len, batch size, 1]
        # trg_rates = [trg len, batch size, 1]
        # trg_lengths = [batch size]

        optimizer.zero_grad()

        output_ids, output_rates = model(choose_norm,src_grid_seqs,src_gps_seqs,src_pro_feas,src_point_feas,src_rids,src_timeids,src_road_feas,src_lengths,  trg_rids, trg_rates,trg_gps_seqs, trg_lengths\
                                         ,constraint_mat,online_features_dict,rn_dict,rid_features_dict)
        
        
        output_rates = output_rates.squeeze(2)
        trg_rids = trg_rids.squeeze(2)
        trg_rates = trg_rates.squeeze(2)

        # output_ids = [trg len, batch size, id one hot output dim]
        # output_rates = [trg len, batch size]
        # trg_rids = [trg len, batch size]
        # trg_rates = [trg len, batch size]

        # rid loss, only show and not bbp
        loss_ids1, recall, precision,f1_score = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths)

        # for bbp
        output_ids_dim = output_ids.shape[-1]
        output_ids = output_ids[1:].reshape(-1, output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
        trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
        # view size is not compatible with input tensor's size and stride ==> use reshape() instead

        loss_train_ids = criterion_ce(output_ids, trg_rids)
        loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
        ttl_loss = loss_train_ids + loss_rates
        
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()
        # print('-----------------------------{}------------------------------------------'.format(i))
        print('total train loss',ttl_loss.item())
        # print_gradient(model)

        epoch_ttl_loss += ttl_loss.item() # total loss
        
        epoch_id1_loss += loss_ids1    #  acc loss
        epoch_recall_loss += recall    # recall loss
        epoch_precision_loss += precision  # precision loss
        epoch_f1_loss += f1_score  # precision loss

        epoch_train_id_loss += loss_train_ids.item()  # id loss
        epoch_rate_loss += loss_rates.item()  # rate loss
        if parameters.test_code:
            break




    return log_vars, epoch_ttl_loss / len(iterator), epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
           epoch_precision_loss / len(iterator),epoch_f1_loss / len(iterator), epoch_rate_loss / len(iterator), epoch_train_id_loss / len(iterator)


def evaluate(model, iterator, rn_dict,  rn_route,\
             online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, parameters):
    model.eval()  # must have this line since it will affect dropout and batch normalization
    device = parameters.device
    epoch_dis_mae_loss = 0
    epoch_dis_rmse_loss = 0

    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_f1_loss = 0

    epoch_rate_loss = 0
    epoch_id_loss = 0 # loss from dl model

    criterion_ce = nn.NLLLoss()
    criterion_reg = nn.MSELoss()

    with torch.no_grad():  # this line can help speed up evaluation
    
        for i, batch in enumerate(iterator):
            src_grid_seqs, src_gps_seqs, src_pro_feas,src_point_feas,src_rids,src_timeids,src_rates,src_road_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths,constraint_mat,choose_norm= batch

          

            constraint_mat = constraint_mat.permute(1, 0, 2).to(device)

            src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
            src_pro_feas = src_pro_feas.float().to(device)
            src_point_feas = src_point_feas.permute(1, 0, 2).to(device)

            src_rids = src_rids.permute(1, 0, 2).long().to(device)
            src_timeids = src_timeids.permute(1, 0, 2).long().to(device)

            src_rates = src_rates.permute(1, 0, 2).to(device)

            src_road_feas = src_road_feas.permute(1, 0, 2).to(device)
            src_gps_seqs = src_gps_seqs.permute(1, 0, 2).to(device)

            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)
            choose_norm= choose_norm.to(device)


            # constraint_mat = [trg len, batch size, id size]
            # src_grid_seqs = [src len, batch size, 2]
            # src_pro_feas = [batch size, feature dim]
            # src_lengths = [batch size]
            # trg_gps_seqs = [trg len, batch size, 2]
            # trg_rids = [trg len, batch size, 1]
            # trg_rates = [trg len, batch size, 1]
            # trg_lengths = [batch size]

            output_ids, output_rates = model(choose_norm,src_grid_seqs,src_gps_seqs,src_pro_feas,src_point_feas,src_rids,src_timeids,src_road_feas,src_lengths,  trg_rids, trg_rates,trg_gps_seqs, trg_lengths\
                                         ,constraint_mat,online_features_dict,rn_dict,rid_features_dict)

            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn_dict, output_ids, output_rates, parameters)
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            # output_ids = [trg len, batch size, id one hot output dim]
            # output_rates = [trg len, batch size]
            # trg_rids = [trg len, batch size]
            # trg_rates = [trg len, batch size]

            # rid loss, only show and not bbp
            loss_ids1, recall, precision,f1_score = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths)
            # distance loss
            dis_mae_loss, dis_rmse_loss, _, _ = check_rn_dis_loss(output_seqs[1:],\
                                                                        output_ids[1:],\
                                                                        output_rates[1:],\
                                                                        trg_gps_seqs[1:],\
                                                                        trg_rids[1:],\
                                                                        trg_rates[1:],\
                                                                        trg_lengths,\
                                                                        rn_route,raw_rn_dict,new2raw_rid_dict,parameters)

            # for bbp
            output_ids_dim = output_ids.shape[-1]
            output_ids = output_ids[1:].reshape(-1,
                                                output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
            trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
            loss_ids = criterion_ce(output_ids, trg_rids)
            # rate loss
            loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
            # loss_rates.size = [(trg len - 1), batch size], --> [(trg len - 1)* batch size,1]
            print('id loss:{},rate loss:{}'.format(loss_ids,loss_rates))
            epoch_dis_mae_loss += dis_mae_loss
            epoch_dis_rmse_loss += dis_rmse_loss

            epoch_id1_loss += loss_ids1
            epoch_recall_loss += recall
            epoch_precision_loss += precision
            epoch_f1_loss += f1_score
            epoch_rate_loss += loss_rates.item()
            epoch_id_loss += loss_ids.item()
            if parameters.test_code:
                break



        return epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
               epoch_precision_loss / len(iterator), epoch_f1_loss / len(iterator),\
               epoch_dis_mae_loss / len(iterator), epoch_dis_rmse_loss / len(iterator), \
               epoch_rate_loss / len(iterator), epoch_id_loss / len(iterator)
    
def test(model, iterator, rn_dict,  rn_route,\
             online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, parameters):
    model.eval()  # must have this line since it will affect dropout and batch normalization
    device = parameters.device
    epoch_dis_mae_loss = 0
    epoch_dis_rmse_loss = 0
    epoch_dis_rn_mae_loss = 0
    epoch_dis_rn_rmse_loss = 0

    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_f1_loss = 0

    epoch_rate_loss = 0
    epoch_id_loss = 0 # loss from dl model

    criterion_ce = nn.NLLLoss()
    criterion_reg = nn.MSELoss()

    with torch.no_grad():  # this line can help speed up evaluation
    
        for i, batch in enumerate(iterator):
            src_grid_seqs, src_gps_seqs, src_pro_feas,src_point_feas,src_rids,src_timeids,src_rates,src_road_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths,constraint_mat,choose_norm= batch

          

            constraint_mat = constraint_mat.permute(1, 0, 2).to(device)

            src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
            src_pro_feas = src_pro_feas.float().to(device)
            src_point_feas = src_point_feas.permute(1, 0, 2).to(device)

            src_rids = src_rids.permute(1, 0, 2).long().to(device)
            src_timeids = src_timeids.permute(1, 0, 2).long().to(device)
            
            src_rates = src_rates.permute(1, 0, 2).to(device)

            src_road_feas = src_road_feas.permute(1, 0, 2).to(device)
            src_gps_seqs = src_gps_seqs.permute(1, 0, 2).to(device)

            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)
            choose_norm= choose_norm.to(device)


            # constraint_mat = [trg len, batch size, id size]
            # src_grid_seqs = [src len, batch size, 2]
            # src_pro_feas = [batch size, feature dim]
            # src_lengths = [batch size]
            # trg_gps_seqs = [trg len, batch size, 2]
            # trg_rids = [trg len, batch size, 1]
            # trg_rates = [trg len, batch size, 1]
            # trg_lengths = [batch size]

            output_ids, output_rates = model(choose_norm,src_grid_seqs,src_gps_seqs,src_pro_feas,src_point_feas,src_rids,src_timeids,src_road_feas,src_lengths,  trg_rids, trg_rates,trg_gps_seqs, trg_lengths\
                                         ,constraint_mat,online_features_dict,rn_dict,rid_features_dict)

            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn_dict, output_ids, output_rates, parameters)
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            # output_ids = [trg len, batch size, id one hot output dim]
            # output_rates = [trg len, batch size]
            # trg_rids = [trg len, batch size]
            # trg_rates = [trg len, batch size]

            # rid loss, only show and not bbp
            loss_ids1, recall, precision,f1_score = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths)
            # distance loss
            dis_mae_loss, dis_rmse_loss, dis_rn_mae_loss, dis_rn_rmse_loss = check_rn_dis_loss(output_seqs[1:],\
                                                                        output_ids[1:],\
                                                                        output_rates[1:],\
                                                                        trg_gps_seqs[1:],\
                                                                        trg_rids[1:],\
                                                                        trg_rates[1:],\
                                                                        trg_lengths,\
                                                                        rn_route,raw_rn_dict,new2raw_rid_dict,parameters,rn_flag=True)

            # for bbp
            output_ids_dim = output_ids.shape[-1]
            output_ids = output_ids[1:].reshape(-1,
                                                output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
            trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
            loss_ids = criterion_ce(output_ids, trg_rids)
            # rate loss
            loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
            # loss_rates.size = [(trg len - 1), batch size], --> [(trg len - 1)* batch size,1]
            print('id loss:{},rate loss:{}'.format(loss_ids,loss_rates))
            epoch_dis_mae_loss += dis_mae_loss
            epoch_dis_rmse_loss += dis_rmse_loss
            epoch_dis_rn_mae_loss += dis_rn_mae_loss
            epoch_dis_rn_rmse_loss += dis_rn_rmse_loss
            epoch_id1_loss += loss_ids1
            epoch_recall_loss += recall
            epoch_precision_loss += precision
            epoch_f1_loss += f1_score
            epoch_rate_loss += loss_rates.item()
            epoch_id_loss += loss_ids.item()
            if parameters.test_code:
                break




        return epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
               epoch_precision_loss / len(iterator), epoch_f1_loss / len(iterator),\
               epoch_dis_mae_loss / len(iterator), epoch_dis_rmse_loss / len(iterator), \
               epoch_dis_rn_mae_loss / len(iterator), epoch_dis_rn_rmse_loss / len(iterator), \
               epoch_rate_loss / len(iterator), epoch_id_loss / len(iterator)

def get_test_time(model, iterator, rn_dict,  rn_route,\
             online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, parameters):
    model.eval()  # must have this line since it will affect dropout and batch normalization
    device = parameters.device
    timer = TimeKeeper()
    memory_before = torch.cuda.memory_allocated(device)
    with torch.no_grad():  # this line can help speed up evaluation
    
        for i, batch in enumerate(iterator):
            src_grid_seqs, src_gps_seqs, src_pro_feas,src_point_feas,src_rids,src_timeids,src_rates,src_road_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths,constraint_mat,choose_norm= batch

          

            constraint_mat = constraint_mat.permute(1, 0, 2).to(device)

            src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
            src_pro_feas = src_pro_feas.float().to(device)
            src_point_feas = src_point_feas.permute(1, 0, 2).to(device)

            src_rids = src_rids.permute(1, 0, 2).long().to(device)
            src_timeids = src_timeids.permute(1, 0, 2).long().to(device)
            src_rates = src_rates.permute(1, 0, 2).to(device)

            src_road_feas = src_road_feas.permute(1, 0, 2).to(device)
            src_gps_seqs = src_gps_seqs.permute(1, 0, 2).to(device)

            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)
            choose_norm= choose_norm.to(device)


            # constraint_mat = [trg len, batch size, id size]
            # src_grid_seqs = [src len, batch size, 2]
            # src_pro_feas = [batch size, feature dim]
            # src_lengths = [batch size]
            # trg_gps_seqs = [trg len, batch size, 2]
            # trg_rids = [trg len, batch size, 1]
            # trg_rates = [trg len, batch size, 1]
            # trg_lengths = [batch size]

            output_ids, output_rates = model(choose_norm,src_grid_seqs,src_gps_seqs,src_pro_feas,src_point_feas,src_rids,src_timeids,src_road_feas,src_lengths,  trg_rids, trg_rates,trg_gps_seqs, trg_lengths\
                                         ,constraint_mat,online_features_dict,rn_dict,rid_features_dict)

            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn_dict, output_ids, output_rates, parameters)
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)

            if parameters.test_code:
                break
        memory_after = torch.cuda.memory_allocated(device)
        memory_used = (memory_after - memory_before)/(1024 * 1024)

        return timer.get_update_time(),memory_used
    
