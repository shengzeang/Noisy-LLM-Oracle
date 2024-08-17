from src.utils import *
import torch 
import os.path as osp
import numpy as np 
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import networkx as nx
from sklearn import cluster


def get_pagerank(edge_index,total_node_number):
    
    edges = [(int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1])]
    nodes = list(range(total_node_number))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    page = torch.tensor(list(pagerank(g).values()))
    return page


def pagerank_select(edge_index ,total_node_number,idx_avilable,total_budget = 140):
    seed_everything(42)
    train_mask = torch.zeros(total_node_number)
    
    page = get_pagerank(edge_index,total_node_number)
    
    page[~idx_avilable]=-999
    _, indices = torch.topk(page, k=total_budget)
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
    
    train_mask= pagerank_select(dataset.edge_index ,len(dataset) ,560)













































































