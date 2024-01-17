import networkx as nx
from rtree import Rtree
from osgeo import ogr
from .spatial_func import SPoint, distance
from .mbr import MBR
import copy
import torch
from tqdm import tqdm
import pandas as pd
class RoadNetwork(nx.DiGraph):
    def __init__(self, g, edge_spatial_idx, node_index):
        super(RoadNetwork, self).__init__(g)
        # entry: eid
        self.edge_spatial_idx = edge_spatial_idx
        # eid -> edge key (start_coord, end_coord)
        self.node_index = node_index
        self.g = g

    def range_query(self, mbr):
        """
        spatial range query
        :param mbr: query mbr
        :return: qualified edge keys
        """
        eids = self.edge_spatial_idx.intersection((mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        return list(eids)

def load_rn_shp(path, is_directed=False):
    # g.nodes are raw edges  g.edge_index  are connectivity between edges
    g = nx.read_gpickle(path)
    edge_spatial_idx = Rtree()
    node_idx = {}
    new_rn_dict = {}

    # edge attrs: eid, length, coords, ...
    for node, data in g.nodes(data=True): 
        coords = [SPoint(value[0],value[1]) for value in data['coords']]
        data['coords'] = coords
        data['length'] = sum([distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
        edge_spatial_idx.insert(data['eid'], data['zone_area'])
        # edge_idx[data['eid']] = data['node_segment']
        node_idx[data['eid']] = data
        eid_value = int(data['eid'])
        new_rn_dict[eid_value ] = {}
        new_rn_dict[eid_value]['coords'] =data['coords']
        new_rn_dict[eid_value]['length'] =data['length']
        new_rn_dict[eid_value]['features'] =data['features']
    print('# self generated of nodes:{}'.format(g.number_of_nodes()))
    return RoadNetwork(g, edge_spatial_idx, node_idx),new_rn_dict

def load_GNN_graph(path):
    # g.nodes are raw edges  g.edge_index  are connectivity between edges
    g = nx.read_gpickle(path)

    # 1 edge_index (直接用rid表示)
    u_edge_index = [u for u,_,_ in g.edges(data=True)]
    v_edge_index = [v for _,v,_ in g.edges(data=True)]
    edge_data = {'s_node': u_edge_index, 'e_node': v_edge_index}
    df_edge = pd.DataFrame(edge_data)
    edge_index = df_edge[["s_node", "e_node"]].to_numpy()
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    print('self generated edge_index finished')
    
    # 2 neighbor_all
    num_nodes = len(g.nodes())
    num_edges = len(edge_index) 
    neighbor_all  =  [torch.arange(num_edges)[edge_index[:,1] == j] for j in tqdm(range(1,num_nodes))]
    max_len = max(map(len,neighbor_all)) 
    neighbor_zero = [num_edges]*max_len 
    neighbor_all = [neighbor_zero]+[list(l) + [num_edges]*(max_len-len(l)) for l in neighbor_all]
    neighbor_all = torch.tensor(neighbor_all).to(torch.int64)
    
    print('# self generated of nodes:{}'.format(g.number_of_nodes()))
    print('# self generated of edges:{}'.format(g.number_of_edges()))
    return edge_index,neighbor_all