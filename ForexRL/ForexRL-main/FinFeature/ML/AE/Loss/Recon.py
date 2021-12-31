import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from FinFeature.ML.DataLoader.FinDataset import FinDataset
from FinFeature.OfflineMarket import OfflineMarket

df = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/rates.csv')
df2 = pd.read_csv(f'/home/z/Desktop/backups/memory_backup/DB/very_smooths.csv')
market_1 = OfflineMarket(df)
market_2 = OfflineMarket(df2)

class Reconstruction:

    def __init__(self,
                 observation,
                 observation_hat
                 ):
        self.obs = observation
        self.obs_hat = observation_hat

    @property
    def mean_square(self):
        return F.mse_loss(self.obs, self.obs_hat)

#
# loss = torch.nn.MSELoss(reduction='mean')
# input = torch.from_numpy(market_1.prices()[:1000,:])
#
# target = torch.from_numpy(market_2.prices()[:1000,:])
#
# output = loss(input, target)
# print(output)
# #print(torch.clip(output, min=3, max=10))
# plt.plot(market_1.prices()[:1000,1])
# plt.plot(market_2.prices()[:1000,1])
# plt.show()

t = torch.randn((10,2,24,60,28))
print(t.size())