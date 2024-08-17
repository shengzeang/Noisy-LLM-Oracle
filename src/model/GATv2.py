from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GCNConv, SAGEConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv as PYGGATConv


class GAT2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = []
        self.bns = []
        self.num_layers = args.num_layers
        self.input_dim  = args.input_dim
        self.hidden_dimension = args.hidden_dimension
        self.num_head = args.num_of_heads
        self.num_out_head =1
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.norm = args.norm
        self.with_bn = True
        
        if self.num_layers == 1:
            self.conv1 = PYGGATConv(self.input_dim, self.hidden_dimension, self.num_head, concat = False, dropout=self.dropout)
        else:
            self.conv1 = PYGGATConv(self.input_dim, self.hidden_dimension, self.num_head, concat = True, dropout=self.dropout)
            self.bns.append(torch.nn.BatchNorm1d(self.hidden_dimension * self.num_head))
        self.layers.append(self.conv1)
        for _ in range(self.num_layers - 2):
            self.layers.append(
                PYGGATConv(self.hidden_dimension * self.num_head, self.hidden_dimension, self.num_head, concat = True, dropout = self.dropout)
            )
            self.bns.append(torch.nn.BatchNorm1d(self.hidden_dimension * self.num_head))

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        if self.num_layers > 1:
            self.layers.append(PYGGATConv(self.hidden_dimension * self.num_head, self.num_classes, heads=self.num_out_head,
                             concat=False, dropout=self.dropout).cuda())
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList(self.bns)

    def forward(self, x ,edge_index):

        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.elu(x)
        return x
