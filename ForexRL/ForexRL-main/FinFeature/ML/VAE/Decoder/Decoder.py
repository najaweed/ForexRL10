import torch
import torch.nn as nn


class MlpDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels):
        super(MlpDecoder, self).__init__()

        self.FC_hidden = nn.Linear(in_channels, hidden_channels)
        self.FC_hidden2 = nn.Linear(hidden_channels, hidden_channels)
        self.FC_output = nn.Linear(hidden_channels, out_channels)

        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = torch.flatten(x)
        # print('decoder input shape ', x.shape)

        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = self.FC_output(h)
        # print('decoder out shape ', x_hat.shape)
        return x_hat
