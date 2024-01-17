import networkx as nx
from rtree import Rtree
from osgeo import ogr
from .spatial_func import SPoint, distance
from .mbr import MBR
import copy

class RoadNetwork(nx.DiGraph):
    def __init__(self, g, edge_spatial_idx, edge_idx,coords_dict):
        super(RoadNetwork, self).__init__(g)
        # entry: eid
        self.edge_spatial_idx = edge_spatial_idx
        # eid -> edge key (start_coord, end_coord)
        self.edge_idx = edge_idx
        self.coords_dict= coords_dict

    def range_query(self, mbr):
        """
        spatial range query
        :param mbr: query mbr
        :return: qualified edge keys
        """
        eids = self.edge_spatial_idx.intersection((mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        return [self.edge_idx[eid] for eid in eids]

    def remove_edge(self, u, v):
        edge_data = self[u][v]
        coords = edge_data['coords']
        mbr = MBR.cal_mbr(coords)
        # delete self.edge_idx[eifrom edge index
        del self.edge_idx[edge_data['eid']]
        # delete from spatial index
        self.edge_spatial_idx.delete(edge_data['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        # delete from graph
        super(RoadNetwork, self).remove_edge(u, v)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        coords = attr['coords']
        mbr = MBR.cal_mbr(coords)
        attr['length'] = sum([distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
        # add edge to edge index
        self.edge_idx[attr['eid']] = (u_of_edge, v_of_edge)
        # add edge to spatial index
        self.edge_spatial_idx.insert(attr['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        # add edge to graph
        super(RoadNetwork, self).add_edge(u_of_edge, v_of_edge, **attr)


def load_rn_shp_route(path,is_directed=True):

    g = nx.read_gpickle(path)
    edge_spatial_idx = Rtree()
    edge_idx = {}
    coords_dict = {}
    # node uses coordinate as key
    # edge uses coordinate tuple as key

    if not is_directed:
        g = g.to_undirected()
    # node attrs: nid, pt, ...

    for n, data in g.nodes(data=True):
        data['pt'] = SPoint(n[1], n[0])
        if 'ShpName' in data:
            del data['ShpName']
    # edge attrs: eid, length, coords, ...
    for k,(u, v, data) in enumerate(g.edges(data=True)):

        coords = [SPoint(value[0],value[1]) for value in data['coords']]
        data['coords'] = coords

        data['length'] = sum([distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
        edge_idx[data['eid']] = (u, v)
        edge_spatial_idx.insert(data['eid'], data['zone_area'])
        coords_dict[data['eid']] = data['coords']
        del data['Wkt']

    print('# of nodes:{}'.format(g.number_of_nodes()))
    print('# of edges:{}'.format(g.number_of_edges()))
    
    return RoadNetwork(g, edge_spatial_idx, edge_idx,coords_dict)
