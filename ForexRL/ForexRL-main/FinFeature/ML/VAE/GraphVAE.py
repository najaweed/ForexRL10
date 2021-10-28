import torch
import torch.nn as nn


class GraphVAE(nn.Module):
    def __init__(self,
                 c_Graph_Encoder: nn.Module,
                 c_Decoder: nn.Module,
                 ):
        super(GraphVAE, self).__init__()
        self.Encoder = c_Graph_Encoder
        self.Decoder = c_Decoder

    @staticmethod
    def latent_reparam(mean, var):
        epsilon = torch.randn_like(var)
        return mean + var * epsilon

    def forward(self, x, edge_index, edge_attr):
        mean, log_var = self.Encoder(x, edge_index, edge_attr)
        latent_space = self.latent_reparam(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(latent_space)

        return x_hat, mean, log_var
