from src.utils import *
import torch 
import os.path as osp
import numpy as np 
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import networkx as nx
from sklearn import cluster
from src.active.pagerank import get_pagerank
from src.active.density import get_density
from torch_geometric.utils import remove_self_loops, scatter, add_self_loops
from torch_sparse import SparseTensor, spmm



def age_select(x,y,edge_index ,total_node_number,idx_avilable,total_budget = 140,alpha=0.33):
    seed_everything(42)
    train_mask = torch.zeros(total_node_number)
    
    num_class=y.max().item() + 1
    aax = compute_norm_aax(x,edge_index,total_node_number)
    aax_density =  get_axx_density(aax,num_class)
    page = get_pagerank(edge_index ,total_node_number)
    
    percentile = (torch.arange(total_node_number, dtype=x.dtype, device=x.device) /total_node_number)
    
    id_sorted = page.argsort(descending=False)
    page[id_sorted] = percentile

    id_sorted = aax_density.argsort(descending=False)
    aax_density[id_sorted] = percentile

    age_score = alpha * aax_density + (1 - alpha) * page
    age_score[~idx_avilable]=-9999
    _, indices = torch.topk(age_score, k=total_budget)
    
    train_mask[indices] = 1
    
    return train_mask.bool()



def get_axx_density(axx,num_of_classes):
    
    model = cluster.KMeans(n_clusters=num_of_classes, init='k-means++', random_state=42)
    model.fit(axx)
    # Calculate density
    centers = model.cluster_centers_
    label = model.predict(axx)
    centers = centers[label]
    dist_map = torch.linalg.norm(axx - centers, dim=1)
    dist_map = torch.tensor(dist_map, dtype=axx.dtype, device=axx.device)
    density = 1 / (1 + dist_map)
    
    return density
    
    
def compute_norm_aax(x, edge_index, num_nodes):
    print("Start computing aax")
    new_edge_index, new_edge_weight = normalize_adj(edge_index, num_nodes)
    # new_edge_index, new_edge_weight = compute_rw_norm_edge_index(edge_index, num_nodes=num_nodes)
    adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
                sparse_sizes=(num_nodes, num_nodes))
    adj_matrix2 = adj.matmul(adj)
    aax = adj_matrix2.matmul(x)
    x = aax.to_dense()
    return x


def compute_rw_norm_edge_index(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float,
                                 device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')

    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    # L = I - A_norm.
    edge_index, tmp = add_self_loops(edge_index, edge_weight,
                                     fill_value=1., num_nodes=num_nodes)
    assert tmp is not None
    edge_weight = tmp
    return edge_index, edge_weight


def normalize_adj(edge_index, num_nodes, edge_weight = None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32,
                                 device=edge_index.device)
    
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight































































































