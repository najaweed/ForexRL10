from FinFeature.ML.VAE.VAE import nnVAE_setup
from FinFeature.ML.VAE.GraphAE import nnGraphAE_setup
from Forex.MetaTrader5.Market import MarketData
from Forex.MetaTrader5.Time import MarketTime

from FinFeature.ML.VAE.DataLoader.FinDataset import FinDataset
from FinFeature.ML.VAE.DataLoader.FinGraphDataset import GraphDataset

from Forex.MetaTrader5.Symbols import currencies, symbols

from torch.optim import SGD


# fin_dataset = FinDataset(fin_features)
# train_loader = fin_dataset.DataLoader(batch_size=30)
# x_dim = fin_dataset.input_channel_size
# setup_vae = nnVAE_setup(x_dim)
mrk = MarketData(MarketTime().time_range)

fin_features = [mrk.percentage_volume[:1000, :]]  # , mrk.log_return[:1000, :]
data_loader = GraphDataset(currencies, symbols, fin_features)

batch_size = 10
train_loader = data_loader.DataLoader(batch_size=batch_size)
x_dim = data_loader.num_node_feature

setup_ae = nnGraphAE_setup(c_input_channel_size=x_dim,
                           c_output_channel_size=data_loader.dim_out_channel,
                           c_number_edge_features=data_loader.number_edges_feature,
                           c_number_nodes=data_loader.number_nodes,
                           c_batch_size=batch_size
                           )
model = setup_ae.model
print(model)
model.train()

optimizer = SGD(model.parameters(), lr=1e-3)

for epoch in range(175):
    overall_loss = 0
    batch_index = None
    for batch_idx, dataset in enumerate(train_loader):
        # TODO change zero_grad to
        #  for param in model.parameters() :
        #       parameter.grad = None
        #   #memory efficiency by Nvidia

        loss = setup_ae.calculate_loss(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.y)

        batch_index = batch_idx
        overall_loss += loss.item()

        optimizer.zero_grad()  # Clear gradients.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / ((batch_index + 1) * 100))

print("Finish!!")
