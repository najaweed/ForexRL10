import torch
import torch.nn as nn

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import TransformerConv


class GraphEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 number_edge_features):
        super().__init__()
        self.edge_dim = number_edge_features
        self.conv1 = TransformerConv(in_channels, out_channels, edge_dim=self.edge_dim, concat=False)
        self.conv2 = TransformerConv(out_channels, out_channels, edge_dim=self.edge_dim, concat=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        return self.conv2(x, edge_index, edge_attr)

print(GraphEncoder(100, 200, 3))
