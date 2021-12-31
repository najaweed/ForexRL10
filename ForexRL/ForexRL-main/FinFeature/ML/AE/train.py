from FinFeature.ML.AE.GraphAE import nnGraphAE_setup
from Forex.MetaTrader5.Market import MarketData
from Forex.MetaTrader5.Time import MarketTime

from FinFeature.ML.DataLoader.FinGraphDataset import GraphDataset

from Forex.MetaTrader5.Symbols import currencies, symbols

import torch
from torch.optim import SGD
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mrk = MarketData(MarketTime().time_range)

fin_features = [mrk.percentage_return[:10000, :]]  # , mrk.log_return[:1000, :]
data_loader = GraphDataset(currencies, symbols, fin_features)

batch_size = 1
train_loader = data_loader.data_loader(batch_size=batch_size)

setup_ae = nnGraphAE_setup(c_input_channel_size=data_loader.num_node_feature,
                           c_output_channel_size=data_loader.dim_out_channel,
                           c_number_edge_features=data_loader.number_edges_feature,
                           c_number_nodes=data_loader.number_nodes,
                           c_batch_size=batch_size
                           )

model = setup_ae.model
print(model)
model.train()
model.cuda()

loss_x = torch.nn.MSELoss()
optimizer = SGD(model.parameters(), lr=1e-3)

for batch_idx, dataset in enumerate(train_loader):
    x = dataset.x.cuda()
    edge_index = dataset.edge_index.cuda()
    edge_attr = dataset.edge_attr.cuda()
    edge_attr = (edge_attr - torch.min(edge_attr)) / (torch.max(edge_attr) - torch.min(edge_attr))

    y = dataset.y.cuda()
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))

    # print(edge_attr)
    # print(edge_attr + 0.1*torch.randn_like(edge_attr))
    x_hat = model(x, edge_index, edge_attr)
    # print(x.shape)
    # print(y.shape)
    loss = loss_x(x_hat, y)
    if loss.item() >10000:
        print(loss.item())
        print(edge_attr.shape,edge_attr)

        print(x_hat.shape,x_hat)
        print(y.shape,y)
        breakpoint()
    print(loss.item(), "\t|", loss_x(y + 0.15 * torch.randn_like(y), y).item())

    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
