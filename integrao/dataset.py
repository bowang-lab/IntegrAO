from torch_geometric.data import InMemoryDataset, Data
from snf.compute import _find_dominate_set
from sklearn.utils.validation import (
    check_array,
    check_symmetric,
    check_consistent_length,
    
)
import networkx as nx
import numpy as np
import torch

# custom dataset
class GraphDataset(InMemoryDataset):

    def __init__(self, neighbor_size, feature, network, transform=None):
        super(GraphDataset, self).__init__('.', transform, None, None)

        neighbor_size = min(int(neighbor_size), network.shape[0])

        # preprocess the input into a pyg graph
        network = _find_dominate_set(network, K=neighbor_size)
        network = check_symmetric(network, raise_warning=False)
        network[network > 0.0] = 1.0
        G = nx.from_numpy_array(network)

        # create edge index from 
        adj = nx.to_scipy_sparse_array(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        data = Data(edge_index=edge_index)
        data.num_nodes = G.number_of_nodes()
        
        # embedding 
        data.x = torch.from_numpy(feature).type(torch.float32)
        
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
   


# custom dataset
class GraphDataset_weight(InMemoryDataset):

    def __init__(self, neighbor_size, feature, network, transform=None):
        super(GraphDataset_weight, self).__init__('.', transform, None, None)

        # preprocess the input into a pyg graph
        network = _find_dominate_set(network, K=neighbor_size)
        network = check_symmetric(network, raise_warning=False)
        
        # Create a binary mask to extract non-zero values for edge weights
        mask = (network > 0.0).astype(float)
        G = nx.from_numpy_array(mask)

        # create edge index from 
        adj = nx.to_scipy_sparse_array(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # Extracting edge weights from the original network using the mask
        edge_weights = network[adj.row, adj.col]

        data = Data(edge_index=edge_index, edge_attr=torch.from_numpy(edge_weights).type(torch.float32))
        data.num_nodes = G.number_of_nodes()
        
        # embedding 
        data.x = torch.from_numpy(feature).type(torch.float32)
        
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
