from pre_processing.spatial_func import SPoint, LAT_PER_METER, LNG_PER_METER, project_pt_to_segment, distance
from pre_processing.mbr import MBR
import math


class CandidatePoint(SPoint):
    def __init__(self, lat, lng, eid, error, offset, rate):
        super(CandidatePoint, self).__init__(lat, lng)
        self.eid = eid
        self.error = error
        self.offset = offset
        self.rate = rate
        self.lat = lat
        self.lng = lng

    def __str__(self):
        return '{},{},{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error, self.offset, self.rate)

    def __repr__(self):
        return '{},{},{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error, self.offset, self.rate)

    def __hash__(self):
        return hash(self.__str__())


def get_candidates(pt,src_rid,rn, search_dist):
    """
    Args:
    -----
    pt: point STPoint()
    rn: road network
    search_dist: in meter. a parameter for HMM_mm. range of pt's potential road
    Returns:
    --------
    candidates: list of potential projected points.
    """
    candidates = None
    
    while True:
        mbr = MBR(pt.lat - search_dist * LAT_PER_METER,
              pt.lng - search_dist * LNG_PER_METER,
              pt.lat + search_dist * LAT_PER_METER,
              pt.lng + search_dist * LNG_PER_METER)
        candidate_nodes = rn.range_query(mbr) 
        # candidate_nodes = all_candidate_nodes
        # for candidate_node in all_candidate_nodes:
        #     road_bearing = rn.node_index[candidate_node]['bearing']
        #     angle_diff = (bearing - road_bearing + math.pi) % (2*math.pi) - math.pi
        #     if abs(angle_diff) >= math.pi/2:
        #         candidate_nodes.remove(candidate_node)

        if len(candidate_nodes) > 0:
            candidate_nodes += src_rid.tolist()
            break
        else:
            search_dist = search_dist+50
    
    candi_pt_list = [cal_candidate_point(pt, rn, candidate_node) for candidate_node in candidate_nodes]

    if len(candi_pt_list) > 0:
        candidates = candi_pt_list
      
    return candidates

def cal_candidate_point(raw_pt, rn, node):
    """
    Get attributes of candidate point
    """

    coords = rn.node_index[node]['coords']  # GPS points in road segment, may be larger than 2
    candidates = [project_pt_to_segment(coords[i], coords[i + 1], raw_pt) for i in range(len(coords) - 1)]
    idx, (projection, coor_rate, dist) = min(enumerate(candidates), key=lambda x: x[1][2])
    # enumerate return idx and (), x[1] --> () x[1][2] --> dist. get smallest error project edge
    offset = 0.0
    for i in range(idx):
        offset += distance(coords[i], coords[i + 1])  # make the road distance more accurately
    offset += distance(coords[idx], projection)  # distance of road start position and projected point
    if rn.node_index[node]['length'] == 0:
        rate = 0
        # print(u, v)
    else:
        rate = offset/rn.node_index[node]['length']  # rate of whole road, coor_rate is the rate of coords.
    return CandidatePoint(projection.lat, projection.lng, rn.node_index[node]['eid'], dist, offset, rate)

# the following are new added 


#  Obtain the specific matching information of the eid corresponding to the trajectory point.
def get_point_candidates(pt,rn,eid):
    """
    Args:
    -----
    pt: point STPoint()
    rn: road network
    search_dist: in meter. a parameter for HMM_mm. range of pt's potential road
    Returns:
    --------
    candidates: list of potential projected points.
    """
    candidates = None
    candidates = cal_candidate_point(pt, rn, eid) 
    return candidates

# get candidate eids between two traj points
def get_inside_candidates_eid(pt,cur_pt,rn, search_dist):
    """
    Args:
    -----
    pt: point STPoint()
    rn: road network
    search_dist: in meter. a parameter for HMM_mm. range of pt's potential road
    Returns:
    --------
    candidates: list of potential projected points.
    """
    candidates = None
    search_dist = search_dist*5
    while True:
        mbr1 = MBR(pt.lat - search_dist * LAT_PER_METER,
              pt.lng - search_dist * LNG_PER_METER,
              pt.lat + search_dist * LAT_PER_METER,
              pt.lng + search_dist * LNG_PER_METER)
        mbr2 = MBR(cur_pt.lat - search_dist * LAT_PER_METER,
              cur_pt.lng - search_dist * LNG_PER_METER,
              cur_pt.lat + search_dist * LAT_PER_METER,
              cur_pt.lng + search_dist * LNG_PER_METER)
        candidate_nodes = rn.range_query(mbr1) +rn.range_query(mbr2) 

        if len(candidate_nodes) > 0:
            break
        else:
            search_dist = search_dist+50
      
    return candidate_nodes

# get   the candidate points eids around a traj point

def get_candidates_eid(pt,rn, search_dist):
    """
    找src 旁边的候选点，初始化图神经网络
    Args:
    -----
    pt: point STPoint()
    rn: road network
    search_dist: in meter. a parameter for HMM_mm. range of pt's potential road
    Returns:
    --------
    candidates: list of potential projected points.
    """
    candidates = None
    

    mbr = MBR(pt.lat - search_dist * LAT_PER_METER,
            pt.lng - search_dist * LNG_PER_METER,
            pt.lat + search_dist * LAT_PER_METER,
            pt.lng + search_dist * LNG_PER_METER)
    all_candidate_nodes = rn.range_query(mbr) 
    candidate_nodes = all_candidate_nodes

    if len(candidate_nodes) > 0:
        return candidate_nodes
    else:
        return [pt.data['candi_pt'].eid]