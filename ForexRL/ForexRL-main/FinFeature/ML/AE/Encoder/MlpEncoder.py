import torch
import torch.nn as nn


class MlpEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MlpEncoder, self).__init__()

        self.FC_input = nn.Linear(in_channels, hidden_channels)
        self.FC_input2 = nn.Linear(hidden_channels, hidden_channels)
        self.FC_mean = nn.Linear(hidden_channels, out_channels)
        self.FC_var = nn.Linear(hidden_channels, out_channels)
        self.Norm_Batch = nn.BatchNorm1d(hidden_channels)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.Norm_Batch(self.FC_input(x)))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        return mean, log_var
