from src.utils import *
import torch 
import os.path as osp
import numpy as np 
from torch_geometric.utils import index_to_mask
import networkx as nx



def random_select( x,y,total_node_number,total_budget = 140):
    seed_everything(42)
    total_num = total_node_number
    total_label_num = total_budget

    train_num = total_label_num
    val_num = 0
    test_num = int(total_num * 0.2)
    
    t_mask, val_mask, test_mask = generate_random_mask(total_node_number, train_num, val_num, test_num)
    
    return t_mask, val_mask, test_mask


def generate_random_mask(total_node_number, train_num, val_num, test_num = -1):
    seed_everything(42)
    random_index = torch.randperm(total_node_number)
    train_index = random_index[:train_num]
    val_index = random_index[train_num:train_num + val_num]
    
    if test_num == -1:
        test_index = random_index[train_num + val_num:]
    else:
        test_index = random_index[train_num + val_num: train_num + val_num + test_num]
        
    return index_to_mask(train_index, total_node_number), index_to_mask(val_index, total_node_number), index_to_mask(test_index, total_node_number)


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
    
    _, val_mask, test_mask= random_select(dataset.feature ,dataset.labels ,len(dataset) ,560)













