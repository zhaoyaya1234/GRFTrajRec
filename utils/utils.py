import pickle
import json
import random
import os
import numpy as np


def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_pkl_data(data, dir, file_name):
    create_dir(dir)
    pickle.dump(data, open(dir + file_name, 'wb'))


def load_pkl_data(dir, file_name):
    '''
    Args:
    -----
        path: path
        filename: file name
    Returns:
    --------
        data: loaded data
    '''
    file = open(dir+file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def save_json_data(data, dir, file_name):
    create_dir(dir)
    with open(dir+file_name, 'w') as fp:
        json.dump(data, fp)


def load_json_data(dir, file_name):
    with open(dir+file_name, 'r') as fp:
        data = json.load(fp)
    return data


def get_min_area(eid,rn_route):
    min_lat = float('inf')
    min_lng = float('inf')
    max_lat =  float('-inf')
    max_lng =  float('-inf')

    for s_eid in eid:
        coordinate = rn_route.coords_dict[s_eid]
        latitudes = [value.lat for value in coordinate] 
        longitudes = [value.lng for value in coordinate]
        
        s_min_lat = min(latitudes)
        s_max_lat = max(latitudes)
        s_min_lng = min(longitudes)
        s_max_lng = max(longitudes)
        
        if s_min_lat < min_lat:
            min_lat = s_min_lat
            
        if s_max_lat > max_lat:
            max_lat = s_max_lat
            
        if s_min_lng < min_lng:
            min_lng = s_min_lng
            
        if s_max_lng > max_lng:
            max_lng = s_max_lng
    print('min_lat :{},min_lng:{},max_lat:{},max_lng:{}'.format(min_lat,min_lng,max_lat,max_lng))
    return (min_lat,min_lng,max_lat,max_lng)

def get_conti_df(df):
    index_list = df.index[df['eid'].notna()].tolist()
    if len(index_list) == len(df):
        return [df]
    else:
        split_dfs = []
        current_group = []
        # 遍历索引列表并将其分成多个子组
        for i, index in enumerate(index_list):
            current_group.append(index)
            # 检查是否是最后一个索引或者是否不连续
            if i == len(index_list) - 1 or index_list[i + 1] != index + 1:
                # 使用iloc来获取当前子组的DataFrame
                sub_df = df.iloc[current_group]
                split_dfs.append(sub_df)
                current_group = []

        split_dfs = [df.reset_index(drop=True) for df in split_dfs]
        return split_dfs
    
def get_nonan_df(traj_dict):
    k = 0 
    traj_dict_map_new = {}
    for value in traj_dict.values():
        df_list = get_conti_df(value)
        for df in df_list:
            traj_dict_map_new[k] = df
            k+=1 
    return traj_dict_map_new