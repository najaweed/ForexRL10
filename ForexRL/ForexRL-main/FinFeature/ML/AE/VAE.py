import torch
import torch.nn as nn
from FinFeature.ML.AE.Encoder.MlpEncoder import MlpEncoder
from FinFeature.ML.AE.Decoder.Decoder import MlpDecoder
from FinFeature.ML.AE.Loss.Recon import Reconstruction
from FinFeature.ML.AE.Loss.Regul import Regularization


class VAE(nn.Module):
    def __init__(self,
                 c_Encoder: nn.Module,
                 c_Decoder: nn.Module,
                 ):
        super(VAE, self).__init__()
        self.Encoder = c_Encoder
        self.Decoder = c_Decoder

    @staticmethod
    def latent_reparam(mean, var):
        epsilon = torch.randn_like(var)
        return mean + var * epsilon

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        latent_space = self.latent_reparam(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(latent_space)

        return x_hat, mean, log_var


class nnVAE_setup:
    def __init__(self,
                 c_input_channel_size,
                 ):
        self.nn_model = VAE
        self.input_channel = c_input_channel_size

    @property
    def model(self):
        return self.nn_model(
            MlpEncoder(in_channels=self.input_channel, hidden_channels=600, out_channels=170),
            MlpDecoder(in_channels=170, hidden_channels=600, out_channels=self.input_channel)
        )

    def calculate_loss(self, dataset):
        x_hat, mean, log_var = self.model(dataset)
        return Reconstruction(dataset, x_hat).mean_square + Regularization(mean, log_var).kl_div
