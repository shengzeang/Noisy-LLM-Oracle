from src.utils import *
import torch 
import os.path as osp
import numpy as np 
from torch_geometric.utils import degree as degree_cal
import networkx as nx
from sklearn import cluster


def degree_select(edge_index ,total_node_number,idx_avilable,total_budget = 140):
    seed_everything(42)
    train_mask = torch.zeros(total_node_number)
    
    row_index = edge_index[0]
    degree = degree_cal(row_index, num_nodes=total_node_number, dtype=torch.long)
    # degree[np.where(train_mask == 0)[0]] = 0
    degree[~idx_avilable]=-9999
    _, indices = torch.topk(degree, k=total_budget)
    
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
    
    train_mask= degree_select(dataset.edge_index ,len(dataset) ,560)






































