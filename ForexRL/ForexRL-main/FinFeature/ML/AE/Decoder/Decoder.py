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

        self.ReLU = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_channels)

        self.norm2 = nn.LayerNorm(out_channels)
    def forward(self, x):
        x = torch.flatten(x)
        # print('decoder input shape ', x.shape)

        h = self.ReLU(self.FC_hidden(x))
        h = self.norm1(h)
        h = self.ReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        x_hat = self.norm2(x_hat)

        # print('decoder out shape ', x_hat.shape)
        return self.ReLU(x_hat)


class MyDecoder(nn.Module):
    def __init__(self, config_decoder):
        super().__init__()
        self.decoders = nn.ModuleList([self.linear_block(conf) for conf in config_decoder])

    def forward(self, x):
        return [decoder(x_in) for decoder, x_in in zip(self.decoders, x)]

    @staticmethod
    def linear_block(config_decoder):
        return nn.Sequential(
            nn.Linear(**config_decoder),
            nn.Sigmoid(),
            nn.Linear(**config_decoder),
            nn.Sigmoid()

        )
