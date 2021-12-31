import networkx as nx
import numpy as np

class StaticFxGraph:
    def __init__(self,
                 c_currencies: list,
                 c_symbols: list,
                 ):
        self.currencies = c_currencies
        self.symbols = c_symbols
        self.G = nx.DiGraph()
        self.add_fx_edges()
        self.list_method_node_attr = self.list_static_method_nodes()

    def add_fx_edges(self):
        for i_sym, sym in enumerate(self.symbols):
            #print(sym)
            #print(self.currencies)
            self.G.add_edge(self.currencies.index(sym[:3]), self.currencies.index(sym[3:6]))

    def list_static_method_nodes(self):
        return [nx.in_degree_centrality(self.G), nx.out_degree_centrality(self.G),
                nx.katz_centrality_numpy(self.G),
                nx.harmonic_centrality(self.G)]  # , nx.clustering(G)]

    def list_static_method_edges(self):
        return [nx.dispersion(self.G)]

    @property
    def node_attrs(self):

        static_node_attrs = np.zeros((len(self.G.nodes), len(self.list_method_node_attr)))
        for i, dict_node_static_attrs in enumerate(self.list_method_node_attr):
            for key, val in dict_node_static_attrs.items():
                static_node_attrs[key, i] = val
                self.G.nodes[key][f'val_{i}'] = val
        return static_node_attrs

    @property
    def static_edges(self):
        edges = [[], []]
        for i_sym, sym in enumerate(self.symbols):
            edges[0].append(self.currencies.index(sym[:3]))
            edges[1].append(self.currencies.index(sym[3:6]))
        return edges
