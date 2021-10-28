import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from FinFeature.ML.VAE.DataLoader.FxGraph import StaticFxGraph


class GraphDataset:
    def __init__(self,
                 c_currencies: list,
                 c_symbols: list,
                 c_edges_features: list[np.ndarray]
                 ):
        self.currencies = c_currencies
        self.symbols = c_symbols
        self.static_nodes = StaticFxGraph(c_currencies, c_symbols).node_attrs
        self.static_edges = StaticFxGraph(c_currencies, c_symbols).static_edges
        self.edges_features = c_edges_features

    def attrs_edges_tick(self, tick_index):
        edges_attrs = np.zeros((len(self.symbols), len(self.edges_features)))
        for i_sym, sym in enumerate(self.symbols):
            for i_fin, fin_feature in enumerate(self.edges_features):
                edges_attrs[i_sym, i_fin] = fin_feature[tick_index, i_sym]
        return edges_attrs

    def DataLoader(self, batch_size=1):
        all_data = []
        for tick in range(self.edges_features[0].shape[0]):
            nodes = torch.tensor(self.static_nodes, dtype=torch.float)

            edge_index = torch.tensor(self.static_edges, dtype=torch.long)

            edge_attr = torch.tensor(self.attrs_edges_tick(tick), dtype=torch.float)
            target_edge_attr = torch.tensor(self.attrs_edges_tick(tick).flatten(), dtype=torch.float)
            all_data.append(Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=target_edge_attr))

        return DataLoader(dataset=all_data, batch_size=batch_size, shuffle=False)

    @property
    def num_node_feature(self):
        return self.static_nodes.shape[1]

    @property
    def number_edges_feature(self):
        return len(self.edges_features)

    @property
    def number_tick(self):
        return self.edges_features[0].shape[0]

    @property
    def number_nodes(self):
        return self.static_nodes.shape[0]

    @property
    def number_edges(self):
        return self.edges_features[0].shape[1]

    @property
    def dim_out_channel(self):
        return len(self.attrs_edges_tick(0).flatten())

from Forex.MetaTrader5.Market import MarketData
from Forex.MetaTrader5.Time import MarketTime
from Forex.MetaTrader5.Symbols import currencies, symbols

# mrk = MarketData(MarketTime().time_range)
# data_volume = mrk.volume[:1000, :]
# data_price = mrk.log_return[:1000, :]
# fin_features = [data_volume, data_price]
# data_loader = GraphDataset(currencies, symbols, fin_features).DataLoader(batch_size=10)
#print(currencies,symbols)
# print(GraphDataset(currencies, symbols, fin_features).number_tick)
#
# print(GraphDataset(currencies, symbols, fin_features).num_node_feature)
#
# print(GraphDataset(currencies, symbols, fin_features).number_edges_feature)
#
# print(GraphDataset(currencies, symbols, fin_features).number_nodes)
# print(GraphDataset(currencies, symbols, fin_features).number_edges)
