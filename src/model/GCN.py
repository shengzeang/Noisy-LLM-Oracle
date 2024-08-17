from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GCNConv, SAGEConv
import torch
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.num_layers = args.num_layers
        self.input_dim  = args.input_dim
        self.hidden_dimension = args.hidden_dimension
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.norm = args.norm
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.num_layers == 1:
            self.convs.append(GCNConv(self.input_dim, self.num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(self.input_dim,self.hidden_dimension, cached=False,
                             normalize=True))
            if self.norm:
                self.norms.append(torch.nn.BatchNorm1d(self.hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(self.num_layers - 2):
                self.convs.append(GCNConv(self.hidden_dimension, self.hidden_dimension, cached=False,
                             normalize=True))
                if self.norm:
                    self.norms.append(torch.nn.BatchNorm1d(self.hidden_dimension))
                else:
                    self.norms.append(torch.nn.Identity())

            self.convs.append(GCNConv(self.hidden_dimension, self.num_classes, cached=False, normalize=True))

    def forward(self, x ,edge_index):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x