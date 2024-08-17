from src.utils import *
import torch 
import os.path as osp
import numpy as np 
from torch_geometric.utils import index_to_mask
import networkx as nx
from sklearn import cluster


def get_density(x, num_of_classes):
    
    model = cluster.KMeans(n_clusters=num_of_classes, init='k-means++', random_state=42)
    model.fit(x)
    # Calculate density
    centers = model.cluster_centers_
    label = model.predict(x)
    centers = centers[label]
    dist_map = torch.linalg.norm(x - centers, dim=1)
    dist_map = torch.tensor(dist_map, dtype=x.dtype, device=x.device)
    density = 1 / (1 + dist_map)
    
    return density

def density_select(x,y,total_node_number,idx_avilable,total_budget = 140):
    
    seed_everything(42)
    train_mask = torch.zeros_like(y)
    num_of_classes = y.max().item() + 1
    
    density = get_density(x,num_of_classes)
    
    density[~idx_avilable]=-9999
    _, indices = torch.topk(density, k=total_budget)
    train_mask[indices] = 1
    
    return train_mask.bool()



if __name__ == '__main__':
    from src.dataset.datasets import BaseDataset
    from src.active.random import random_select
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-ml', default='gcn', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='nc', type=str, help='node classification or link prediction')
    parser.add_argument('--dataset', '-d', default='cora', type=str, help='name of datasets')
    parser.add_argument('--device', '-de', default=2, type=int, help='-1 means cpu')
    args = parser.parse_args(args=[])
    dataset=BaseDataset(args)
    
    train_mask= density_select(dataset.feature ,dataset.labels ,len(dataset) ,560)





































