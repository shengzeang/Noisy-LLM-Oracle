from src.utils import *
import torch
import numpy as np
from torch_sparse import SparseTensor, spmm
from torch_geometric.utils import remove_self_loops, scatter, add_self_loops
import copy
import time

from src.lib.utils import update_reliability_single_node, get_max_reliable_influ_node


def dma(args, features, labels, edge_index, total_node_number, idx_avilable, total_budget=140, oracle_acc=1, th=0.05, prior=None, batch_size=5):
    seed_everything(42)

    reliability_list = torch.ones(total_node_number)
    all_idx = torch.arange(total_node_number)
    adj_matrix2 = compute_adj2(edge_index, total_node_number).to_dense()
    norm_aax = compute_norm_aax(features, edge_index, total_node_number)
    similarity_feature = norm_aax

    idx_train = []
    idx_available = all_idx[idx_avilable].tolist()
    idx_available_temp = copy.deepcopy(idx_available)
    activated_node = torch.ones(total_node_number)
    count = 0
    iter = 0
    train_class = {}
    time1 = time.time()

    while True:
        max_ral_node, max_activated_node, max_activated_num = get_max_reliable_influ_node(
            torch.tensor(idx_available_temp), activated_node, reliability_list, th, adj_matrix2, is_first=(iter == 0))

        idx_train.append(max_ral_node)
        idx_available_temp.remove(max_ral_node)
        node_label = labels[max_ral_node].item()
        if node_label in train_class:
            train_class[node_label].append(max_ral_node)
        else:
            train_class[node_label] = list()
            train_class[node_label].append(max_ral_node)
        count += 1

        activated_node = activated_node - max_activated_node
        activated_node = torch.clamp(activated_node, min=0)

        if count % batch_size == 0:
            time0 = time.time()
            update_reliability(args, adj_matrix2, idx_train, labels, total_node_number,
                                                reliability_list, oracle_acc, similarity_feature, th, is_first=(iter == 0))
            time1 = time.time()
            iter += 1

        if count >= total_budget or max_activated_num <= 0:
            # print(max_activated_num)
            break

    train_mask = torch.zeros(total_node_number)
    train_mask[idx_train] = 1

    return train_mask.bool()


def update_reliability(args, adj_matrix2, idx_used, labels, num_node, reliability_list, oracle_acc, similarity_feature, th, is_first=None):

    alpha = oracle_acc
    visited = torch.zeros(num_node)
    if is_first:
        reliability_list[idx_used] = alpha
        # visited[idx_used] = 1

    num_class = labels.max().item()+1
    num_feature = similarity_feature.shape[1]
    sim_label = []
    sim_label_cor = 0.
    for i in range(num_class):
        sim_label.append(
            (sum(args.labels_sim[i])-args.labels_sim[i][i])/(num_class-1))
        sim_label_cor += args.labels_sim[i][i]
    sim_label_cor /= num_class
    sim_label = torch.tensor(sim_label, dtype=torch.float32)
    activated_node = torch.zeros(num_node)

    cur_similarity_feature = torch.mm(similarity_feature[idx_used], similarity_feature.t())
    dis_range = torch.max(cur_similarity_feature) - torch.min(cur_similarity_feature)
    cur_similarity_feature = (cur_similarity_feature -
                          torch.min(cur_similarity_feature))/dis_range

    for i, node in enumerate(idx_used):
        idx_used_mask = torch.zeros(num_node)
        idx_used_mask[idx_used] = 1

        update_reliability_single_node(reliability_list, num_node, node, adj_matrix2[node].to_dense(), torch.tensor(labels), idx_used_mask, 
                                       cur_similarity_feature[i], args.labels_sim, sim_label, num_class, visited, sim_label_cor)

    # normalize the realiability list
    dis_range = torch.max(reliability_list) - torch.min(reliability_list)
    reliability_list = (reliability_list -
                            torch.min(reliability_list))/dis_range


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


def normalize_adj(edge_index, num_nodes, edge_weight=None):
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


def compute_adj2(edge_index, num_nodes):
    print("Computing adj2")
    n_edge_index, n_edge_weight = compute_rw_norm_edge_index(
        edge_index, num_nodes=num_nodes)
    adj = SparseTensor(row=n_edge_index[0], col=n_edge_index[1], value=n_edge_weight,
                       sparse_sizes=(num_nodes, num_nodes))
    return adj


def compute_norm_aax(x, edge_index, num_nodes):
    print("Start computing aax")
    new_edge_index, new_edge_weight = compute_rw_norm_edge_index(edge_index, num_nodes=num_nodes)
    adj = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight,
                       sparse_sizes=(num_nodes, num_nodes))
    aax = adj.matmul(x)
    aax = adj.matmul(aax)
    x = aax.to_dense()

    return x


def compute_sim(norm_aax, num_nodes):
    print("compute sim")

    similarity_feature = torch.mm(norm_aax, norm_aax.t())
    dis_range = torch.max(similarity_feature) - torch.min(similarity_feature)
    similarity_feature = (similarity_feature -
                          torch.min(similarity_feature))/dis_range

    return similarity_feature
