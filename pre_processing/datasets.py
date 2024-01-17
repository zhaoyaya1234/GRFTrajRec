import random
from tqdm import tqdm
import os
from chinese_calendar import is_holiday
import numpy as np
import torch
from pre_processing.spatial_func import distance
from pre_processing.trajectory import get_tid, Trajectory
from utils.parse_traj import ParseMMTraj
from utils.save_traj import SaveTraj2MM
from utils.utils import create_dir
from models.model_utils import toseq, get_constraint_mask,get_choose_norm
from map_matching.candidate_point import get_candidates_eid

def get_win_trajs_df(df, win_size):
    len_df = len(df)
    if len_df < win_size:
        return [df]

    num_win = len_df // win_size
    last_traj_len = len_df % win_size + 1
    new_dfs = []
    for w in range(num_win + 1):
        # if last window is large enough then split to a single DataFrame
        if w == num_win and last_traj_len > 15:
            tmp_df = df.iloc[win_size * w - 1:]
        # elif last window is not large enough then merge to the last DataFrame
        if w == num_win - 1 and last_traj_len <= 15:
            ind = 0 
            if win_size * w - 1 > 0:
                ind = win_size*w-1
            tmp_df = df.iloc[ind:]
        if (w == num_win - 1 and last_traj_len > 15) or w < num_win - 1:
            tmp_df = df.iloc[max(0, (win_size * w - 1)):win_size * (w + 1)]

        new_dfs.append(tmp_df)
    return new_dfs

def get_last_dict(dict_list,args):
    df_list = [list(data_dict.values())[0] for data_dict in dict_list]
    print('raw_length',len(df_list))
    final_dict = {}
    ss = 0 
    for df in df_list:
        new_dfs = get_win_trajs_df(df,args.win_size)
        for new_df in new_dfs:
            final_dict[ss] = new_df.reset_index(drop=True)
            ss+=1
    return final_dict
def data_augment_dict(df_dict):
    random_seed = 42  # 你可以选择任何整数作为种子值
    random.seed(random_seed)
    df_list = [list(data_dict.values())[0] for data_dict in df_dict]

    aug_list = []
    for df in df_list:
        for start_row in range(15):
            result_df = df.iloc[start_row::15]
            result_df.reset_index(drop=True, inplace=True) 
            if len(result_df)>10:
                aug_list.append(result_df) 
    selected_dfs = sorted(aug_list, key=len, reverse=True)[:len(aug_list)//2]
    keep_count = int(len(selected_dfs)*(2/3))
    selected_dfs = random.sample(selected_dfs, keep_count)
    aug_dict = [{k:selected_dfs[k]} for k in range(len(selected_dfs))]
    return aug_dict
def split_data(traj_input_path, output_dir,args,random_seed=42):
    """
    split original data to train, valid and test datasets
    """
    random.seed(random_seed)
    create_dir(output_dir)
    train_data_dir = output_dir + 'train_data'
    create_dir(train_data_dir)
    val_data_dir = output_dir + 'valid_data'
    create_dir(val_data_dir)
    test_data_dir = output_dir + 'test_data'
    create_dir(test_data_dir)
    
    trg_trajs = np.load(traj_input_path,allow_pickle=True).item()
    trg_trajs = [{key:trg_trajs[key]} for key in trg_trajs.keys()]
    ttl_lens = len(trg_trajs)

    test_inds = random.sample(range(ttl_lens), int(ttl_lens * 0.1))  # 10% as test data
    tmp_inds = [ind for ind in range(ttl_lens) if ind not in test_inds]
    val_inds = random.sample(tmp_inds, int(ttl_lens * 0.2))  # 20% as validation data
    train_inds = [ind for ind in tmp_inds if ind not in val_inds]  # 70% as training data
 
    train_data = [trg_trajs[j] for j in train_inds]
    val_data = [trg_trajs[j] for j in val_inds]
    test_data = [trg_trajs[j] for j in test_inds]
    if args.city == 'yancheng':

        train_data = data_augment_dict(train_data)

        val_data = data_augment_dict(val_data)

        test_data = data_augment_dict(test_data)


    print('----------------prepare for HMM--------------------------------')
    
    train_data_dict = get_last_dict(train_data,args)
    val_data_dict = get_last_dict(val_data,args)
    test_data_dict = get_last_dict(test_data,args)
    
    np.save('{}/train_dict.npy'.format(train_data_dir),train_data_dict)
    print("HMM target traj train len: ", len(train_data_dict))
    np.save('{}/val_dict.npy'.format(val_data_dir),val_data_dict)
    print("HMM target traj valid  len: ", len(val_data_dict))
    np.save('{}/test_dict.npy'.format(test_data_dir),test_data_dict)
    print("HMM target traj test len: ", len(test_data_dict))
    
    parser = ParseMMTraj()
    test_data = parser.parse(test_data)
    np.save('{}/test.npy'.format(test_data_dir),test_data)
    print("target traj test len: ", len(test_data))

    val_data  = parser.parse(val_data )
    np.save('{}/val.npy'.format(val_data_dir),val_data)
    print("target traj val len: ", len(val_data))

    train_data = parser.parse(train_data)
    np.save('{}/train.npy'.format(train_data_dir),train_data)
    print("target traj train len: ", len(train_data))
    
    


class Dataset(torch.utils.data.Dataset):
    """
    customize a dataset for PyTorch
    """

    def __init__(self, trajs_dir, mbr,online_features_dict,raw2new_rid_dict,\
                        parameters,seed_value):
        self.mbr = mbr  # MBR of all trajectories
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span  # time interval between two consecutive points.
        self.online_features_flag = parameters.online_features_flag
        self.src_grid_seqs, self.src_gps_seqs, self.src_pro_fea,self.src_point_fea,self.src_rid,self.src_timeid,self.src_rate,self.src_road_fea,self.choose_rid_all = [], [], [],[],[],[],[],[],[]
        self.trg_gps_seqs, self.trg_rids, self.trg_rates = [], [], []
        self.new_tids = []
        self.rn = parameters.rn
        # above should be [num_seq, len_seq(unpadded)]
        random.seed(seed_value)
        self.get_data(trajs_dir,online_features_dict,raw2new_rid_dict,parameters.win_size, 
             parameters.ds_type, parameters.keep_ratio)
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.src_grid_seqs)

    def __getitem__(self, index):
        """Generate one sample of data"""
        
        src_grid_seq = self.src_grid_seqs[index]
        src_gps_seq = self.src_gps_seqs[index]
        src_rid = self.src_rid[index]
        src_timeid = self.src_timeid[index]
        src_rate = self.src_rate[index]
        
        trg_gps_seq = self.trg_gps_seqs[index]
        trg_rid = self.trg_rids[index]
        trg_rate = self.trg_rates[index]

        src_grid_seq = self.add_token(src_grid_seq)
        src_gps_seq = self.add_token(src_gps_seq)
        src_rid = self.add_token(src_rid)
        src_timeid = self.add_token(src_timeid)
        src_rate = self.add_token(src_rate)

        trg_gps_seq = self.add_token(trg_gps_seq)
        trg_rid = self.add_token(trg_rid)
        trg_rate = self.add_token(trg_rate)
        

        # src_point_fea   features for every traj point  
        # src_road_fea    features for every single road segment
        # src_pro_fea    features  for a  ovearall
        src_point_fea = torch.tensor(self.add_token(self.src_point_fea[index]))
        src_road_fea = torch.tensor(self.add_token(self.src_road_fea[index]))
        src_pro_fea = torch.tensor(self.src_pro_fea[index])

        choose_rid = torch.tensor(self.choose_rid_all[index])
    

        return src_grid_seq, src_gps_seq, src_pro_fea,src_point_fea,src_rid,src_timeid,src_rate,src_road_fea, trg_gps_seq, trg_rid, trg_rate,choose_rid

    def add_token(self, sequence):
        """
        Append start element(sos in NLP) for each sequence. And convert each list to tensor.
        """
        new_sequence = []
        dimension = len(sequence[0])
        start = [0] * dimension  # pad 0 as start of rate sequence
        new_sequence.append(start)
        new_sequence.extend(sequence)
        new_sequence = torch.tensor(new_sequence)
        return new_sequence
    def get_data(self,trajs_dir,online_features_dict,raw2new_rid_dict, win_size, \
                      ds_type, keep_ratio):
        # parser = ParseMMTraj()
        # trajs = parser.parse(trajs_dir)
        trajs =np.load(trajs_dir,allow_pickle=True)
        for traj in tqdm(trajs):
            new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, \
            ls_grid_seq_ls, ls_gps_seq_ls, features_ls,point_features_ls,ls_rid_ls,ls_timeid_ls,ls_src_rate,ls_rid_features,choose_rid_ls = self.parse_traj(traj,online_features_dict,raw2new_rid_dict,win_size, ds_type, keep_ratio)
            if new_tid_ls is not None:
                self.new_tids.extend(new_tid_ls)

                self.trg_gps_seqs.extend(mm_gps_seq_ls)
                self.trg_rids.extend(mm_eids_ls)
                self.trg_rates.extend(mm_rates_ls)

                self.src_grid_seqs.extend(ls_grid_seq_ls)
                self.src_gps_seqs.extend(ls_gps_seq_ls)
                self.src_pro_fea.extend(features_ls)
                self.src_point_fea.extend(point_features_ls)

                self.src_rid.extend(ls_rid_ls)
                self.src_timeid.extend(ls_timeid_ls)
                self.src_rate.extend(ls_src_rate)
                self.src_road_fea.extend(ls_rid_features)
                self.choose_rid_all.extend(choose_rid_ls)

                assert len(new_tid_ls) == len(mm_gps_seq_ls) == len(mm_eids_ls) == len(mm_rates_ls) 
      

        assert len(self.new_tids) == len(self.trg_gps_seqs) == len(self.trg_rids) == len(self.trg_rates) == \
               len(self.src_gps_seqs) == len(self.src_grid_seqs) == len(self.src_pro_fea) == len(self.src_point_fea)== len(self.src_rid) ==len(self.src_timeid)== len(self.src_road_fea) == len(self.choose_rid_all), \
        'The number of source and target sequence must be equal.'

    def parse_traj(self,traj,online_features_dict,raw2new_rid_dict,win_size, ds_type, keep_ratio):
        """
        Split traj based on length.
        Preprocess ground truth (map-matched) Trajectory(), get gps sequence, rid list and rate list.
        Down sample original Trajectory(), get ls_gps, ls_grid sequence and profile features
        Args:
        -----
        traj:
            Trajectory()
        win_size:
            window size of length for a single high sampling trajectory
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_steps element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        --------
        new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, ls_grid_seq_ls, ls_gps_seq_ls, features_ls
        """
        new_trajs = self.get_win_trajs(traj, win_size)
        choose_rid_ls = []
        new_tid_ls = []
        mm_gps_seq_ls, mm_eids_ls, mm_rates_ls = [], [], []
        ls_grid_seq_ls, ls_gps_seq_ls, features_ls,point_features_ls,ls_rid_ls,ls_timeid_ls,ls_src_rate,ls_rid_features  = [], [], [],[],[],[],[],[]

        for tr in new_trajs:
            tmp_pt_list = tr.pt_list
            new_tid_ls.append(tr.tid)

            # get target sequence

            mm_gps_seq, mm_eids, mm_rates = self.get_trg_seq(tmp_pt_list,raw2new_rid_dict)
            if mm_eids is None:
                return None, None, None, None, None, None, None

            # get source sequence

            ds_pt_list = self.downsample_traj(tmp_pt_list, ds_type, keep_ratio)

            ls_grid_seq, ls_gps_seq,point_features, ls_src_eids,ls_src_timeids,ls_src_rate_s,rid_features,hours, ttl_t,choose_rid = self.get_src_seq(ds_pt_list,online_features_dict,raw2new_rid_dict)
            features = self.get_pro_features(tr.oid,ds_pt_list, hours)
            # check if src and trg len equal, if not return none
            if len(mm_gps_seq) != ttl_t:
                print(len(mm_gps_seq))
                print(ttl_t)
                print(ds_pt_list)
                return None, None, None, None, None, None, None
            
            mm_gps_seq_ls.append(mm_gps_seq)
            mm_eids_ls.append(mm_eids)
            mm_rates_ls.append(mm_rates)

            ls_grid_seq_ls.append(ls_grid_seq)
            ls_gps_seq_ls.append(ls_gps_seq)
            features_ls.append(features)
            point_features_ls.append(point_features)
            ls_rid_ls.append(ls_src_eids)
            ls_timeid_ls.append(ls_src_timeids)
            ls_src_rate.append(ls_src_rate_s)

            ls_rid_features.append(rid_features)
            choose_rid_ls.append(choose_rid)

        return new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, \
            ls_grid_seq_ls, ls_gps_seq_ls, features_ls,point_features_ls,ls_rid_ls,ls_timeid_ls,ls_src_rate,ls_rid_features,choose_rid_ls
      

    def get_win_trajs(self, traj, win_size):
        pt_list = traj.pt_list
        len_pt_list = len(pt_list)
        if len_pt_list < win_size:
            return [traj]

        num_win = len_pt_list // win_size
        last_traj_len = len_pt_list % win_size + 1
        new_trajs = []
        ss = 0 
        for w in range(num_win+1):
            # if last window is large enough then split to a single trajectory
            if w == num_win and last_traj_len > 15:
                tmp_pt_list = pt_list[win_size * w - 1:]
            # elif last window is not large enough then merge to the last trajectory
            if w == num_win - 1 and last_traj_len <= 15:
                # fix bug, when num_win = 1
                ind = 0
                if win_size * w - 1 > 0:
                    ind = win_size * w - 1
                tmp_pt_list = pt_list[ind:]
            if (w == num_win - 1 and last_traj_len > 15) or w < num_win - 1:
                tmp_pt_list = pt_list[max(0, (win_size * w - 1)):win_size * (w + 1)]

            # -1 to make sure the overlap between two trajs

            new_traj = Trajectory(traj.oid, str(traj.tid)+"_"+str(ss), tmp_pt_list)
            # print('dist',new_traj.get_distance())
            new_trajs.append(new_traj)
            ss+=1
        return new_trajs

    def get_trg_seq(self, tmp_pt_list,raw2new_rid_dict):
        mm_gps_seq = []
        mm_eids = []
        mm_rates = []
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            if candi_pt is None:
                return None, None, None
            else:
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
                mm_eids.append([raw2new_rid_dict[candi_pt.eid]])  # keep the same format as seq
                mm_rates.append([candi_pt.rate])
        return mm_gps_seq, mm_eids, mm_rates

    def get_search_distance(self,pt_list):
        """
        Get geographical distance of a trajectory (pt_list)
        sum of two adjacent points
        meters
        """
        dist = []
        pre_pt = pt_list[0]
        for pt in pt_list[1:]:
            tmp_dist = distance(pre_pt, pt)
            dist.append(tmp_dist)
            pre_pt = pt
        search_distance = np.zeros(len(pt_list))
        for i in range(len(pt_list)):
            if i == 0:
                search_distance[i]=dist[i]/2
            elif i == len(pt_list)-1:
                search_distance[i]=dist[-1]/2
            else:
                search_distance[i]=max(dist[i-1],dist[i])/2
        return list(search_distance)
    def get_src_seq(self, ds_pt_list,online_features_dict,raw2new_rid_dict):
        hours = []
        ls_grid_seq = []
        ls_gps_seq = []
        point_features = []
        ls_src_eids = []
        ls_src_timeids = []

        ls_src_rate_s = []
        rid_features = []
        first_pt = ds_pt_list[0]
        last_pt = ds_pt_list[-1]
        time_interval = self.time_span
        ttl_t = self.get_noramlized_t(first_pt, last_pt, time_interval)
        for ds_pt in ds_pt_list: 
            hours.append(ds_pt.time.hour) 
            t = self.get_noramlized_t(first_pt, ds_pt, time_interval)
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            locgrid_xid, locgrid_yid = self.gps2grid(ds_pt, self.mbr, self.grid_size)
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])
            ls_src_rate_s.append([ds_pt.data['candi_pt'].rate])
            point_features.append(ds_pt.data['traj_features'])  
            
            ls_src_eids.append([raw2new_rid_dict[ds_pt.data['candi_pt'].eid]])
            ls_src_timeids.append([int(ds_pt.time.timestamp())])
            rid_features.append(online_features_dict[raw2new_rid_dict[ds_pt.data['candi_pt'].eid]][4:])
        
        search_distance = self.get_search_distance(ds_pt_list)
        choose_rid = [candidate for k,ds_pt in enumerate(ds_pt_list) for  candidate in get_candidates_eid(ds_pt,self.rn, search_distance[k])]
        return ls_grid_seq, ls_gps_seq,point_features, ls_src_eids,ls_src_timeids,ls_src_rate_s,rid_features,hours, ttl_t,choose_rid 

    '''
    改轨迹特征
    '''
    def get_pro_features(self,uid, ds_pt_list, hours):
        holiday = is_holiday(ds_pt_list[0].time)*1
        day = ds_pt_list[0].time.day
        hour = {'hour': np.bincount(hours).max()}  # find most frequent hours as hour of the trajectory
        features = [uid]+ self.one_hot(hour) + [holiday]
        return features
    
    
    '''
    get road network rid and features
    '''
    
    def gps2grid(self, pt, mbr, grid_size):
        """
        mbr:
            MBR class.
        grid size:
            int. in meter
        """
        LAT_PER_METER = 8.993203677616966e-06
        LNG_PER_METER = 1.1700193970443768e-05
        lat_unit = LAT_PER_METER * grid_size
        lng_unit = LNG_PER_METER * grid_size
        
        max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
        max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1
        
        lat = pt.lat
        lng = pt.lng
        locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
        locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1
        
        return locgrid_x, locgrid_y
    
    
    def get_noramlized_t(self, first_pt, current_pt, time_interval):
        """
        calculate normalized t from first and current pt
        return time index (normalized time)
        """

        t = int(1+((current_pt.time - first_pt.time).seconds/time_interval))

        return t

    @staticmethod
    def get_distance(pt_list):
        dist = 0.0
        pre_pt = pt_list[0]
        for pt in pt_list[1:]:
            tmp_dist = distance(pre_pt, pt)
            dist += tmp_dist
            pre_pt = pt
        return dist


    @staticmethod
    def downsample_traj(pt_list, ds_type, keep_ratio):
        """
        Down sample trajectory
        Args:
        -----
        pt_list:
            list of Point()
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_stepth element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        -------
        traj:
            new Trajectory()
        """
        assert ds_type in ['uniform', 'random','block'], 'only `uniform` or `random` is supported'

        old_pt_list = pt_list.copy()
        start_pt = old_pt_list[0]
        end_pt = old_pt_list[len(pt_list)-1]

        if ds_type == 'uniform':
            if (len(old_pt_list) - 1) % int(1 / keep_ratio) == 0:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)]
            else:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)] + [end_pt]
            
        elif ds_type == 'random': 
            sampled_inds = sorted(\
                random.sample(range(1, len(old_pt_list) - 1), int((len(old_pt_list) - 2) * keep_ratio)))
            new_pt_list = [start_pt] + list(np.array(old_pt_list)[sampled_inds]) + [end_pt]

        elif ds_type == 'block':
            num_points_to_keep = int((len(old_pt_list) - 2) * keep_ratio)
            num_points_in_block = len(old_pt_list) - 2 - num_points_to_keep

            # Randomly select the starting point of the block within the valid range
            # random.randint 
            block_start = random.randint(1, len(old_pt_list) - num_points_in_block-1)

            # Calculate the ending point of the block
            block_end = block_start + num_points_in_block

            # Create the new trajectory by removing the selected block
            new_pt_list = list(np.array(old_pt_list)[:block_start]) + list(np.array(old_pt_list)[block_end:])
        else:
            print('ds_type error')
        
        return new_pt_list


    

    @staticmethod
    def one_hot(data):
        one_hot_dict = {'hour': 24, 'weekday': 7}
        for k, v in data.items():
            encoded_data = [0] * one_hot_dict[k]
            encoded_data[v - 1] = 1
        return encoded_data
    


def collate_fn(data,rn, args):
    """
    Args:
    -----
    data: list of tuple (src_seq, src_pro_fea, trg_seq, trg_rid, trg_rate), from dataset.__getitem__().
        - src_seq: torch tensor of shape (?,2); variable length.
        - src_pro_fea: torch tensor of shape (1,64) # concatenate all profile features
        - trg_seq: torch tensor of shape (??,2); variable length.
        - trg_rid: torch tensor of shape (??); variable length.
        - trg_rate: torch tensor of shape (??); variable length.
    Returns:
    --------
    src_grid_seqs:
        torch tensor of shape (batch_size, padded_length, 3)
    src_gps_seqs:
        torch tensor of shape (batch_size, padded_length, 3).
    src_pro_feas:
        torch tensor of shape (batch_size, feature_dim) unnecessary to pad
    src_point_feas:
        torch tensor of shape (batch_size, padded_length,point_feature_dim) unnecessary to pad
    src_rids:
        torch tensor of shape (batch_size, padded_length, 1).
    src_road_feas:
        torch tensor of shape (batch_size, padded_length, road_feature_dim).
        
    src_lengths:
        list of length (batch_size); valid length for each padded source sequence.

    
    trg_seqs:
        torch tensor of shape (batch_size, padded_length, 2).
    trg_rids:
        torch tensor of shape (batch_size, padded_length, 1).
    trg_rates:
        torch tensor of shape (batch_size, padded_length, 1).

    trg_lengths:
        list of length (batch_size); valid length for each padded target sequence.
    """

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by source sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_grid_seqs, src_gps_seqs, src_pro_feas,src_point_feas,src_rids,src_timeids,src_rates,src_road_feas, trg_gps_seqs, trg_rids, trg_rates,choose_rids =zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_grid_seqs, src_lengths = merge(src_grid_seqs)
    src_gps_seqs, src_lengths = merge(src_gps_seqs)
    src_pro_feas = torch.tensor([list(src_pro_fea) for src_pro_fea in src_pro_feas])
    src_point_feas,_ =  merge(src_point_feas)
    src_point_feas = torch.tensor(src_point_feas)

    src_rids, _ = merge(src_rids)
    src_rids = src_rids.long()

    src_timeids, _ = merge(src_timeids)
    src_timeids = src_timeids.long()

    src_rates,_ = merge(src_rates)
    src_road_feas,_ =  merge(src_road_feas)
    src_road_feas = torch.tensor(src_road_feas)

    trg_gps_seqs, trg_lengths = merge(trg_gps_seqs)
    trg_rids, _ = merge(trg_rids)
    trg_rids = trg_rids.long()
    trg_rates, _ = merge(trg_rates)

    constraint_mat = get_constraint_mask(src_grid_seqs, src_gps_seqs,src_rids,src_lengths, trg_lengths, rn, args)
    choose_norm = get_choose_norm(choose_rids,args)
    return src_grid_seqs, src_gps_seqs, src_pro_feas,src_point_feas,src_rids,src_timeids,src_rates,src_road_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths,constraint_mat,choose_norm
    
