from map_matching.candidate_point import get_point_candidates
import re
from datetime import datetime
import pandas as pd
import numpy as np
import time
from pre_processing.trajectory import Trajectory, STPoint
from pre_processing.spatial_func import SPoint, LAT_PER_METER, LNG_PER_METER, project_pt_to_segment, distance
from pre_processing.mbr import MBR
from tqdm import tqdm
import math
from map_matching.hmm.hmm_map_matcher import TIHMMMapMatcher
from utils.parse_traj import ParseRawTraj
def truncate_to_sublists(lst):
    sublists = []
    sublist = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            sublist.append(lst[i])
        else:
            sublists.append(sublist)
            sublist = [lst[i]]
    sublists.append(sublist)  # 添加最后一个子列表
    longest_sublist = max(sublists, key=len)
    return longest_sublist
def get_candi(rn_route,trg_trajs):

    map_model = TIHMMMapMatcher(rn_route)
    parser = ParseRawTraj() 
    raw_trajs = trg_trajs
    trg_trajs = [{key:trg_trajs[key]} for key in trg_trajs.keys()]
    trg_trajs = parser.parse(trg_trajs)
    traj_dict_map = {}
    for traj in tqdm(trg_trajs):
        key = int(traj.tid)
        map_traj = map_model.match(traj)
        assert len(traj.pt_list) == len(map_traj.pt_list)
        stpoint_list = [stpoint for stpoint in map_traj.pt_list]
        try:
            eid =[stpoint.data['candi_pt'].eid for stpoint in stpoint_list]
            proj_lat =[stpoint.data['candi_pt'].lat for stpoint in stpoint_list]
            proj_lng =[stpoint.data['candi_pt'].lng for stpoint in stpoint_list]
            error = [stpoint.data['candi_pt'].error for stpoint in stpoint_list]
            offset =[stpoint.data['candi_pt'].offset for stpoint in stpoint_list]
            rate = [stpoint.data['candi_pt'].rate  for stpoint in stpoint_list]
            time = [stpoint.time  for stpoint in stpoint_list]
            traj_dict_map[key] = raw_trajs[key][["uid","longitude","latitude","timestamp","speed","timestamp_long"]]
            traj_dict_map[key][["eid","proj_lat","proj_lng","error" ,"offset" ,"rate","time"]] = pd.DataFrame({"eid":eid,"proj_lat":proj_lat,"proj_lng":proj_lng,"error":error,"offset":offset ,"rate" :rate,"time":time})
            result = (traj_dict_map[key]["timestamp_long"] == traj_dict_map[key]["time"]).all() or (traj_dict_map[key]["timestamp"] == traj_dict_map[key]["time"]).all()
            assert result == True
            traj_dict_map[key] = traj_dict_map[key].drop(columns=['time'])
        except:
            pass
    return traj_dict_map

def get_candi_proba(rn_route,trg_trajs):

    map_model = TIHMMMapMatcher(rn_route)
    parser = ParseRawTraj() 

    raw_trajs = trg_trajs
    trg_trajs = [{key:trg_trajs[key]} for key in trg_trajs.keys()]
    trg_trajs = parser.parse(trg_trajs)
    traj_dict_map = {}
    for traj in tqdm(trg_trajs):
        key = int(traj.tid)
        map_traj = map_model.match(traj)
        assert len(traj.pt_list) == len(map_traj.pt_list)
        stpoint_list = [stpoint for stpoint in map_traj.pt_list]
        try:
            eid =[stpoint.data['candi_pt'].eid for stpoint in stpoint_list]
            proj_lat =[stpoint.data['candi_pt'].lat for stpoint in stpoint_list]
            proj_lng =[stpoint.data['candi_pt'].lng for stpoint in stpoint_list]
            error = [stpoint.data['candi_pt'].error for stpoint in stpoint_list]
            offset =[stpoint.data['candi_pt'].offset for stpoint in stpoint_list]
            rate = [stpoint.data['candi_pt'].rate  for stpoint in stpoint_list]
            time = [stpoint.time  for stpoint in stpoint_list]
            traj_dict_map[key] = raw_trajs[key][["uid","longitude","latitude","timestamp","speed","timestamp_long"]]
            traj_dict_map[key][["eid","proj_lat","proj_lng","error" ,"offset" ,"rate","time"]] = pd.DataFrame({"eid":eid,"proj_lat":proj_lat,"proj_lng":proj_lng,"error":error,"offset":offset ,"rate" :rate,"time":time})
            result = (traj_dict_map[key]["timestamp_long"] == traj_dict_map[key]["time"]).all() or (traj_dict_map[key]["timestamp"] == traj_dict_map[key]["time"]).all()
            assert result == True
            traj_dict_map[key] = traj_dict_map[key].drop(columns=['time'])
        except:
            index = []
            for k in range(len(stpoint_list)):
                try:
                    stpoint_list[k].data['candi_pt']==None
                except:
                    index.append(k)

            eid =[stpoint_list[k].data['candi_pt'].eid for k in index]
            proj_lat =[stpoint_list[k].data['candi_pt'].lat for k in index]
            proj_lng =[stpoint_list[k].data['candi_pt'].lng for k in index]
            error = [stpoint_list[k].data['candi_pt'].error for  k in index]
            offset =[stpoint_list[k].data['candi_pt'].offset for  k in index]
            rate = [stpoint_list[k].data['candi_pt'].rate  for  k in index]
            time = [stpoint_list[k].time  for k in index]
            traj_dict_map[key] = raw_trajs[key][["uid","longitude","latitude","timestamp","speed","timestamp_long"]]
            map_values = {"eid":eid,"proj_lat":proj_lat,"proj_lng":proj_lng,"error":error,"offset":offset ,"rate" :rate,"time":time}
            # 将值赋给DataFrame的指定行、多列
            traj_dict_map[key].loc[index, ['eid', 'proj_lat', 'proj_lng', 'error', 'offset', 'rate', 'time']] = map_values.values()
    return traj_dict_map