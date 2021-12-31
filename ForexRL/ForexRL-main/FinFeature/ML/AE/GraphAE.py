import torch
import torch.nn as nn
from FinFeature.ML.AE.Decoder.Decoder import MlpDecoder

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

class GraphVAE(nn.Module):
    def __init__(self,
                 c_Graph_Encoder: nn.Module,
                 c_Decoder: nn.Module,
                 ):
        super(GraphVAE, self).__init__()
        self.Encoder = c_Graph_Encoder
        self.Decoder = c_Decoder

    def forward(self, x, edge_index, edge_attr):
        latent_space = self.Encoder(x, edge_index, edge_attr)
        return self.Decoder(latent_space)


class nnGraphAE_setup:
    def __init__(self,
                 c_input_channel_size,
                 c_output_channel_size,
                 c_number_edge_features,
                 c_number_nodes,
                 c_batch_size,
                 ):
        self.nn_model = GraphVAE
        self.num_node_features = c_input_channel_size
        self.out_channels = c_output_channel_size
        self.num_edge_features = c_number_edge_features
        self.num_nodes = c_number_nodes
        self.batch_size = c_batch_size

        self.loss = torch.nn.MSELoss().cuda()

    @property
    def model(self):
        return self.nn_model(
            GraphEncoder(in_channels=self.num_node_features,
                         out_channels=128,
                         number_edge_features=self.num_edge_features),
            MlpDecoder(in_channels=128*self.batch_size*self.num_nodes,
                       hidden_channels=256,
                       out_channels=self.out_channels*self.batch_size)
        )

    def calculate_loss(self, x, edge_index, edge_attr, y):
        x_hat = self.model(x, edge_index, edge_attr)
        return self.loss(y, x_hat)#Reconstruction(y, x_hat).mean_square
