import os
from src.dataset.datasets import BaseDataset
from src.active.random import random_select
from src.active.RIM import rim_select
import argparse
from src.active.density import *
from src.train import train_pipeline
from src.active.density import get_density
from src.active.degree import degree_select
from src.active.pagerank import pagerank_select
from src.active.AGE import age_select
from src.active.gpart import *
from src.active.dma import dma
import math

from ogb.nodeproppred import PygNodePropPredDataset


def node_active_learning(args, dataset, idx_avilable):
    print("#------------------------------------node_select--------------------------------#")
    labels = dataset.y.squeeze(1)
    num_classes = labels.max().item() + 1
    _, val_mask, test_mask = random_select(
        dataset.x, labels, dataset.num_nodes, args.budget)

    if args.active == "density":
        train_mask = density_select(dataset.x, labels, 
            dataset.num_nodes, idx_avilable, args.budget)
    elif args.active == "degree":
        train_mask = degree_select(dataset.edge_index, 
            dataset.num_nodes, idx_avilable, args.budget)
    elif args.active == "age":
        train_mask = age_select(dataset.x, labels, dataset.edge_index, 
            dataset.num_nodes, idx_avilable, args.budget)
    elif args.active == "gpart":
        partitions = gpart_preprocess(
            dataset.x, dataset.edge_index, max_part=args.max_part)
        indices = gpart_select(dataset.x, dataset.edge_index, partitions, args.compensation,
                               max_part=args.max_part, num_centers=args.num_centers, total_budget=args.budget)
        train_mask = torch.zeros(dataset.num_nodes)
        train_mask[indices] = 1
        train_mask = train_mask.bool()
    elif args.active == "rim":
        train_mask = rim_select(dataset.x, args.pre_labels, dataset.edge_index, 
            dataset.num_nodes, idx_avilable, args.budget, th=0.2, oracle_acc=args.oracle_acc)
    elif args.active == "pagerank":
        train_mask = pagerank_select(dataset.edge_index, 
            dataset.num_nodes, idx_avilable, args.budget)
    elif args.active == "dma":
        train_mask = dma(args, dataset.x, args.pre_labels, dataset.edge_index, 
            dataset.num_nodes, idx_avilable, args.budget, oracle_acc=args.oracle_acc, th=args.th)

    test_mask = ~train_mask
    print("select_nodes_num:", train_mask.sum(),
          " test_nodes_num:", test_mask.sum())

    return train_mask, val_mask, test_mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-ml', default='gcn', type=str)
    parser.add_argument('--active', '-ac', default='density', type=str)
    parser.add_argument('--budget', default=120, type=int)
    parser.add_argument('--task', '-t', default='nc', type=str)
    parser.add_argument('--dataset', '-d', default='cora', type=str)
    parser.add_argument('--oracle_acc', '-ora', default=1, type=float)
    parser.add_argument('--device', '-de', default=0, type=int)
    parser.add_argument('--seeds', '-se', default=10, type=int)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-6)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--max_part", type=int, default=7)
    parser.add_argument("--compensation", type=float, default=1)
    parser.add_argument("--num_centers", type=int, default=1)
    parser.add_argument("--th", type=float, default=0.01)
    parser.add_argument('--num-threads', type=int, default=int(os.cpu_count()))
    args = parser.parse_args()
    dataset = BaseDataset(args)
    
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/dataset')
    graph = dataset[0]

    args.num_layers = 2
    args.hidden_dimension = 64
    args.dropout = 0.5
    args.num_of_heads = 8
    args.norm = None
    args.input_dim = graph.num_features
    args.num_classes = graph.y.max().item() + 1
    args.labels_sim = torch.load(osp.join(
        'data/', f"mistral8x7b_{args.dataset}_sim_simple.pt"), map_location='cpu').detach().numpy()
    args.labels_sim = torch.tensor(args.labels_sim, dtype=torch.float32)
    args.pre_labels = graph.y.squeeze(1).clone()
    args.budget = 20 * args.num_classes

    idx_avilable = args.pre_labels >= 0
    train_mask, val_mask, test_mask = node_active_learning(
        args, graph, idx_avilable)
    mask = (train_mask, val_mask, test_mask)
    this_train_acc, this_test_acc, this_best_test_acc = train_pipeline(
        args, mask, graph, args.pre_labels)

    print(f"Train Accuracy: {np.mean(this_train_acc) * 100:.2f} ± {np.std(this_train_acc) * 100:.2f}")
    print(f"Test Accuracy: {np.mean(this_test_acc)* 100:.2f} ± {np.std(this_test_acc)* 100:.2f}")
