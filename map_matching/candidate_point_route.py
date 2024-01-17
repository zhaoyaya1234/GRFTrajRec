from pre_processing.spatial_func import SPoint, LAT_PER_METER, LNG_PER_METER, project_pt_to_segment, distance
from pre_processing.mbr import MBR
import math

'''
This code implements a function get_candidates, whose purpose is to find the nearest road segment from the road network based on a given point pt, and to return as candidate points the projection points on the segment that are within a certain distance from the point.

Another function cal_candidate_point is used, which calculates the nearest projection point to a given point on a given road segment and returns the relevant attributes of the point, including the coordinates of the projection point, the error, the offset on the road segment and the ratio of the road segment.

Specific implementation details can be seen in the function comments and variable names in the code, for example, pt means point, rn means road network, search_dist means search radius, eid means the ID of the road section, error means distance error and so on.
'''

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


def get_candidates(pt, rn, search_dist):
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
    mbr = MBR(pt.lat - search_dist * LAT_PER_METER,
              pt.lng - search_dist * LNG_PER_METER,
              pt.lat + search_dist * LAT_PER_METER,
              pt.lng + search_dist * LNG_PER_METER)
    candidate_edges = rn.range_query(mbr)  # list of edges (two nodes/points)
    if len(candidate_edges) > 0:
        candi_pt_list = [cal_candidate_point(pt, rn, candidate_edge) for candidate_edge in candidate_edges]
        # refinement
        candi_pt_list = [candi_pt for candi_pt in candi_pt_list if candi_pt.error <= search_dist]
        if len(candi_pt_list) > 0:
            candidates = candi_pt_list
    return candidates


def cal_candidate_point(raw_pt, rn, edge):
    """
    Get attributes of candidate point
    """
    u, v = edge
    coords = rn[u][v]['coords']  # GPS points in road segment, may be larger than 2
    candidates = [project_pt_to_segment(coords[i], coords[i + 1], raw_pt) for i in range(len(coords) - 1)]
    idx, (projection, coor_rate, dist) = min(enumerate(candidates), key=lambda x: x[1][2])
    # enumerate return idx and (), x[1] --> () x[1][2] --> dist. get smallest error project edge
    offset = 0.0
    for i in range(idx):
        offset += distance(coords[i], coords[i + 1])  # make the road distance more accurately
    offset += distance(coords[idx], projection)  # distance of road start position and projected point
    if rn[u][v]['length'] == 0:
        rate = 0
        # print(u, v)
    else:
        rate = offset/rn[u][v]['length']  # rate of whole road, coor_rate is the rate of coords.
    return CandidatePoint(projection.lat, projection.lng, rn[u][v]['eid'], dist, offset, rate)