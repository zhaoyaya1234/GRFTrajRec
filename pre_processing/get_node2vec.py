import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random
import networkx as nx
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec
import time
import datetime
from tqdm import tqdm

random.seed(1953)

def get_Node2Vec_model(num_nodes,edge_index): # num_nodes :id_size
    config = yaml.safe_load(open('pre_processing/config.yaml'))
    feature_size = config["feature_size"]
    walk_length = config["node2vec"]["walk_length"]
    context_size = config["node2vec"]["context_size"]
    walks_per_node = config["node2vec"]["walks_per_node"]
    p = config["node2vec"]["p"]
    q = config["node2vec"]["q"]
    Node2Vec_model = Node2Vec(
    edge_index,
    embedding_dim=feature_size,
    walk_length=walk_length,
    context_size=context_size,
    walks_per_node=walks_per_node,
    num_negative_samples=1,
    p=p,
    q=q,
    sparse=True,
    num_nodes=num_nodes
)
    return Node2Vec_model

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch(model, loader, optimizer,epoch):
    # Training with epoch iteration
    last_loss = 1
    print("Training node embedding with node2vec...")
    for i in range(epoch):  # loss 
        loss = train(model, loader, optimizer)
        print('Epoch: {0} \tLoss: {1:.4f}'.format(i, loss))
        if abs(last_loss - loss) < 1e-5:
            print('stop training')
            break
        else:
            last_loss = loss

@torch.no_grad()
def save_embeddings(model, num_nodes,save_path):
    model.eval()
    node_features = model(torch.arange(num_nodes)).cpu().numpy()
    np.save("{}/node_features_GNN.npy".format(save_path), node_features)
    return

def load_GNN_netowrk(edge_index,save_path):
    """
    load road network from file with Pytorch geometric data object
    :param dataset: the city name of road network
    :return: Pytorch geometric data object of the graph
    """
    node_embedding_path = save_path + "/node_features_GNN.npy"

    node_embeddings = np.load(node_embedding_path,allow_pickle=True)
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)

    embeddings = node_embeddings
    print("embeddings shape: ", embeddings.shape)
    print("edge_index shape: ", edge_index.shape)

    road_network = Data(x=embeddings, edge_index=edge_index)

    return road_network

def load_netowrk(edge_index,save_path):
    """
    load road network from file with Pytorch geometric data object
    :param dataset: the city name of road network
    :return: Pytorch geometric data object of the graph
    """
    node_embedding_path = save_path + "/node_features.npy"

    node_embeddings = np.load(node_embedding_path,allow_pickle=True)
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)

    embeddings = node_embeddings
    print("embeddings shape: ", embeddings.shape)
    print("edge_index shape: ", edge_index.shape)

    road_network = Data(x=embeddings, edge_index=edge_index)

    return road_network