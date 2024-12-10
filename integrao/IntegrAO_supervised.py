import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch

from torch_geometric.nn import GraphSAGE


class IntegrAO(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, pred_n_layer=2, pred_act="softplus", num_classes=None):
        super(IntegrAO, self).__init__()
        self.in_channels = in_channels # this is an array since we have multiple domains
        self.hidden_channels = hidden_channels
        self.output_dim = out_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.pred_n_layer=2,
        self.pred_act="softplus",

        num = len(in_channels)
        feature = []
        for i in range(num):
            model_sage = GraphSAGE(
                in_channels=self.in_channels[i], 
                hidden_channels=self.hidden_channels, 
                num_layers=self.num_layers, 
                out_channels=self.output_dim,
                project=False,)
            
            feature.append(model_sage)

        self.feature = nn.ModuleList(feature)

        self.feature_show = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.output_dim, self.output_dim),
        )

        self.pred_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2 ),
            nn.BatchNorm1d(self.output_dim // 2),
            nn.LeakyReLU(0.1, True),
            # nn.Softplus(),
            nn.Linear(self.output_dim // 2, self.num_classes),
        )

        print(self)


    def get_sample_ids_for_domain(self, domain):
        return self.sample_ids[domain]


    def forward(self, x_dict, edge_index_dict, domain_sample_ids):
        z_all = {}
        z_sample_dict = {}
        for domain in x_dict.keys():
            z = self.feature[domain](x_dict[domain], edge_index_dict[domain])
            z = self.feature_show(z)
            z_all[domain] = z

            # Let's assume that your samples have unique identifiers and you
            # can extract these identifiers for each domain
            sample_ids = domain_sample_ids[domain]

            # Go through each sample and its corresponding vector
            for sample_id, vector in zip(sample_ids, z):
                # If the sample's vectors haven't been recorded, create a new list
                if sample_id not in z_sample_dict:
                    z_sample_dict[sample_id] = []

                # Append the new vector to the list of vectors
                z_sample_dict[sample_id].append(vector)

        # Now, average the vectors for each sample
        z_avg = {}
        for sample_id, vectors in z_sample_dict.items():
            # Stack all vectors along a new dimension and calculate the mean
            z_avg[sample_id] = torch.stack(vectors).mean(dim=0)

        sorted_list = z_avg.items() # sorted(z_avg.items())
        z_avg_list = [z_avg for _, z_avg in sorted_list]
        z_id_list = [z_id for z_id, _ in sorted_list]
        z_avg_tensor = torch.stack(z_avg_list)

        output = self.pred_head(z_avg_tensor)
        return z_all, z_avg_tensor, output, z_id_list

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
