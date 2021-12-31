import torch
from torch import nn
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import time

from FinFeature.ML.DataLoader.FinDataset import FinDataset
from FinFeature.OfflineMarket import OfflineMarket

df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/smooths.csv')
df2 = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/rates.csv')
market_1 = OfflineMarket(df)
market_2 = OfflineMarket(df2)

xx_input = market_1.tensor_rates_for_cnn((1, 1, 1, 1, 50000))
xxx_input = market_2.tensor_rates_for_cnn((1, 1, 1, 1, 50000))

print(np.hstack((xxx_input,xx_input)).shape)
xx_input = np.moveaxis(xx_input, 0, 1)
x_input = np.copy(xx_input[:3, ...])

data_loader = FinDataset(xx_input).data_loader(batch_size=3)


def ae_config():
    config1 = {
        "encoder": [],
        "decoder": [],
    }

    in_out_channels = [3, 32, 128, 128]
    kernels = [20, 10, 4, 3]
    strides = [1, 1, 1, 1]
    paddings = [1, 1, 0, 0]
    shape_outs = [x_input.shape]
    for i in range(len(in_out_channels) - 1):
        conf_enc = dict(in_channels=in_out_channels[i],
                        out_channels=in_out_channels[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        padding=paddings[i],
                        )

        # H_out = np.floor((shape_outs[i][2] + 2 * paddings[i][0] - 1 - (kernels[i][0] - 1)) / strides[i][0] + 1)
        # W_out = np.floor((shape_outs[i][3] + 2 * paddings[i][1] - 1 - (kernels[i][1] - 1)) / strides[i][1] + 1)
        # shape_outs.append((shape_outs[0][0], in_out_channels[i + 1], H_out, W_out))
        L_out = np.floor((shape_outs[i][2] + 2 * paddings[i] - 1 - (kernels[i] - 1)) / strides[i] + 1)
        shape_outs.append((shape_outs[0][0], in_out_channels[i + 1], L_out))

        config1['encoder'].append(conf_enc)

    de_in_out_channels = [128, 128, 64, 32, 16, 3]
    de_kernels = [1, 1, 5, 10, 23, 30]
    de_strides = [1, 1, 1, 1, 1, 1]
    de_paddings = [0, 0, 1, 1, 2, 0]

    for i in range(len(de_in_out_channels) - 1):
        conf_dec = dict(in_channels=de_in_out_channels[i],
                        out_channels=de_in_out_channels[i + 1],
                        kernel_size=de_kernels[i],
                        stride=de_strides[i],
                        padding=de_paddings[i],
                        )

        config1['decoder'].append(conf_dec)
    return config1


class MyEncoder(nn.Module):
    def __init__(self,
                 config_encoder,
                 # config_latent,
                 ):
        super().__init__()

        self.conv_blocks = nn.Sequential(*[self.conv_block(conf) for conf in config_encoder])
        # self.latents = nn.Sequential(*[self.latent_block(conf) for conf in config_latent])

    def forward(self, x):
        return self.conv_blocks(x)  # self.latents(x.view(-1))

    @staticmethod
    def conv_block(config_encoder):
        return nn.Sequential(
            nn.Conv1d(**config_encoder),
            nn.BatchNorm1d(config_encoder['out_channels']),
            nn.ReLU()
        )

    @staticmethod
    def latent_block(config_latent):
        return nn.Sequential(
            nn.Linear(**config_latent),
            nn.Tanh()
        )


class MyDecoder(nn.Module):
    def __init__(self,
                 config_decoder,
                 # config_reconstruction,
                 ):
        super().__init__()
        # self.config_reconstruction = config_reconstruction
        self.decoders = nn.Sequential(*[self.conv_block(conf) for conf in config_decoder])

    def forward(self, x):
        return self.decoders(x)

    @staticmethod
    def conv_block(config_encoder):
        return nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(**config_encoder),
            nn.BatchNorm1d(config_encoder['out_channels']),
            nn.ReLU()
        )

    @staticmethod
    def linear_block(config_decoder):
        return nn.Sequential(
            nn.Linear(**config_decoder),
            nn.Tanh()
        )


# from torchinfo import summary


class AutoEncoder(nn.Module):
    def __init__(self,
                 config,
                 ):
        super(AutoEncoder, self).__init__()
        self.Encoder = MyEncoder(config['encoder'])
        self.Decoder = MyDecoder(config['decoder'])

    def forward(self, x):
        x = self.Encoder(x)
        x_hat = self.Decoder(x)
        return x_hat


config_1 = ae_config()
# for d, data in enumerate(data_loader):
#     print('shape input', data.shape)
#     AutoEncoder(config_1)(data)
#     #encode = MyEncoder(config_1['encoder']).conv_blocks(data)
#     #summary(MyEncoder(config_1['encoder']).conv_blocks, data.shape)
#     #print('shape out encoder', encode.shape)
#     #decode = MyDecoder(config_1['decoder']).decoders(encode)
#     summary(AutoEncoder(config_1), data.shape)
#     print('shape out decoder', AutoEncoder(config_1)(data).shape)
#
#     if d == 0:
#         break


model = AutoEncoder(config_1).cuda()
#print(model)
model.train()
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, )


s_time = time.time()
for i, data in enumerate(data_loader):
    data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    # print(data)
    x_input = data.cuda()
    x_hat1 = model(x_input)


    loss = criterion(x_hat1, x_input)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(model.Encoder(x_input))
    print(criterion(data + 0.05 * torch.randn_like(data), data).data)
    print(loss.data)
    print(criterion(data + 0.2 * torch.randn_like(data), data).data)
    print(criterion(torch.rand_like(data), data).data)
    print("-------")

    # if i == 0:
    #     break

e_time = time.time()
print(e_time - s_time)
torch.save(model.state_dict(), "model.pth")
model.cpu()
model.eval()

criterion = nn.MSELoss(reduction='sum')

with torch.no_grad():
    for data in data_loader:
        data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))

        # print(data)
        x_input = data
        x_hat1 = model(x_input)

        loss = criterion(x_hat1, data)
        print(loss.data)
        print(model.Encoder(x_input).shape)
