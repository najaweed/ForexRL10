from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self,
                 input_x: np.ndarray):
        self.input_x = input_x.astype('float32')

    def __len__(self):
        return self.input_x.shape[0]

    def __getitem__(self, index):
        return self.input_x[index, ...]


class FinDataset:
    def __init__(self,
                 c_fin_features:np.ndarray,
                 ):
        self.fin_features = c_fin_features
        #self.fin_features = np.hstack(self.fin_features)

    def data_loader(self, batch_size=1 , num_workers = 0):
        return DataLoader(dataset=MyDataset(self.fin_features), batch_size=batch_size,num_workers=num_workers , drop_last=True)

    @property
    def input_channel_size(self):
        return self.fin_features.shape[1]

    @property
    def number_tick(self):
        return self.fin_features.shape[0]



