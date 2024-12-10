import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from torch_geometric.nn import GraphSAGE


class IntegrAO(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(IntegrAO, self).__init__()
        self.in_channels = in_channels # this is an array since we have multiple domains
        self.hidden_channels = hidden_channels
        self.output_dim = out_channels
        self.num_layers = num_layers

        num = len(in_channels)
        feature = []

        for i in range(num):
            model_sage = GraphSAGE(
                in_channels=self.in_channels[i], 
                hidden_channels=self.hidden_channels, 
                num_layers=self.num_layers, 
                out_channels=self.output_dim,
                project=False,
)
            
            feature.append(model_sage)

        self.feature = nn.ModuleList(feature)

        self.feature_show = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, x_dict, edge_index_dict):
        z_all = {}
        for domain in x_dict.keys():
            z = self.feature[domain](x_dict[domain], edge_index_dict[domain])
            z = self.feature_show(z)
            z_all[domain] = z
        
        return z_all
