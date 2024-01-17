# Reference: https://github.com/huiminren/tptk/blob/master/common/trajectory.py

import re
from datetime import datetime
import pandas as pds
import numpy as np
import time
from pre_processing.trajectory import Trajectory, STPoint
from map_matching.candidate_point import CandidatePoint
from tqdm import tqdm



class ParseTraj:
    """
    ParseTraj is an abstract class for parsing trajectory.
    It defines parse() function for parsing trajectory.
    """
    def __init__(self):
        pass

    def parse(self, input_path):
        """
        The parse() function is to load data to a list of Trajectory()
        """
        pass

class ParseRawTraj(ParseTraj):
    """
    Parse map matched GPS points to trajectories list. No extra data preprocessing
    """
    def __init__(self):
        super().__init__()
    def parse(self, all_traj):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list. list of trajectories. trajs contain input_path file's all gps points
        """
        # all_traj = np.load(input_path,allow_pickle=True)
        tid_list = [u for i in range(len(all_traj)) for u,_ in all_traj[i].items()]
        content_list = [v for i in range(len(all_traj)) for _,v in all_traj[i].items()]
        time_format = '%Y-%m-%d %H:%M:%S'
        trajs = []
        i = 0
        for single_traj in tqdm(content_list):

            oid = int(single_traj['uid'][0])
            tid = tid_list[i]
            # 单点轨迹特征
            try:
                pt_list = single_traj.apply(lambda x:STPoint(x.latitude,x.longitude,datetime.strptime(str(x.timestamp),time_format)), axis=1)
            except:
                pt_list = single_traj.apply(lambda x:STPoint(x.latitude,x.longitude,datetime.strptime(str(x.timestamp_long),time_format)), axis=1)
           
            pt_list = [pt_list[i] for i in range(len(pt_list)) ]
            traj = Trajectory(oid, tid, pt_list)
            trajs.append(traj)
            i+=1
        return trajs
         
class ParseMMTraj(ParseTraj):
    """
    Parse map matched GPS points to trajectories list. No extra data preprocessing
    """
    def __init__(self):
        super().__init__()

    def parse(self, all_traj):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list. list of trajectories. trajs contain input_path file's all gps points
        """
        # all_traj = np.load(input_path,allow_pickle=True)
        tid_list = [u for i in range(len(all_traj)) for u,_ in all_traj[i].items()]
        content_list = [v for i in range(len(all_traj)) for _,v in all_traj[i].items()]
        time_format = '%Y-%m-%d %H:%M:%S'
        trajs = []
        i = 0
        for single_traj in tqdm(content_list):
            oid = int(single_traj['uid'][0])
            tid = tid_list[i]
            # 单点轨迹特征
            try:
                pt_list = single_traj.apply(lambda x:STPoint(x.latitude,x.longitude,datetime.strptime(str(x.timestamp),time_format), {'candi_pt':CandidatePoint(x.proj_lat, x.proj_lng, x.eid, x.error, x.offset, x.rate),'traj_features':[x.speed]}), axis=1)
            except:
                pt_list = single_traj.apply(lambda x:STPoint(x.latitude,x.longitude,datetime.strptime(str(x.timestamp_long),time_format), {'candi_pt':CandidatePoint(x.proj_lat, x.proj_lng, x.eid, x.error, x.offset, x.rate),'traj_features':[x.speed]}), axis=1)

            
            pt_list = [pt_list[i] for i in range(len(pt_list)) ]
            traj = Trajectory(oid, tid, pt_list)
            trajs.append(traj)
            i+=1
        return trajs
         