import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import numpy as np
import pandas as pd
import networkx as nx
import time
import os
from snf.compute import _find_dominate_set

from integrao.IntegrAO_supervised import IntegrAO
from integrao.dataset import GraphDataset
import torch_geometric.transforms as T

def tsne_loss(P, activations):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = activations.size(0)
    alpha = 1
    eps = 1e-12
    sum_act = torch.sum(torch.pow(activations, 2), 1)
    Q = (
        sum_act
        + sum_act.view([-1, 1])
        - 2 * torch.matmul(activations, torch.transpose(activations, 0, 1))
    )
    Q = Q / alpha
    Q = torch.pow(1 + Q, -(alpha + 1) / 2)
    Q = Q * autograd.Variable(1 - torch.eye(n), requires_grad=False).to(device)
    Q = Q / torch.sum(Q)
    Q = torch.clamp(Q, min=eps)
    C = torch.log((P + eps) / (Q + eps))
    C = torch.sum(P * C)
    return C


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 100 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 100))
    lr = max(lr, 1e-3)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_model(net, device, restore):
    if restore is not None and os.path.exits(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    else:
        pass

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.to(device)

    return net


def P_preprocess(P):
    # Make sure P-values are set properly
    np.fill_diagonal(P, 0)  # set diagonal to zero
    P = P + np.transpose(P)  # symmetrize P-values
    P = P / np.sum(P)  # make sure P-values sum to one
    # P = P * 4.0  # early exaggeration
    P = np.maximum(P, 1e-12)
    return P

def _load_pre_trained_weights(model, model_path, device):
    try:
        state_dict = torch.load(
            os.path.join(model_path, "model.pth"), map_location=device
        )
        # model.load_state_dict(state_dict)
        model.load_my_state_dict(state_dict)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")

    return model

def tsne_p_deep_classification(dicts_commonIndex, dict_sampleToIndexs, dict_original_order, data, clf_labels, model_path=None, P=np.array([]), neighbor_size=20, embedding_dims=50, alighment_epochs=1000, num_classes=2):
    """
    Runs t-SNE on the dataset in the NxN matrix P to extract embedding vectors
    to no_dims dimensions.
    """
    
    # Check inputs
    if isinstance(embedding_dims, float):
        print("Error: array P should have type float.")
        return -1
    if round(embedding_dims) != embedding_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    print("Starting supervised fineting!")
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_channels = 128 # TODO: change to using ymal file
    dataset_num = len(P)
    feature_dims = []
    transform = T.Compose([
        T.ToDevice(device), 
    ])

    # clf_labels is a dataframe
    labels = torch.from_numpy(clf_labels.values.flatten()).long().to(device)

    x_dict = {}
    edge_index_dict = {}
    for i in range(dataset_num):
        # preprocess the inputs into PyG graph format
        dataset = GraphDataset(neighbor_size, data[i], P[i], transform=transform)
        x_dict[i] = dataset[0].x
        edge_index_dict[i] = dataset[0].edge_index
        
        feature_dims.append(np.shape(data[i])[1])
        print("Dataset {}:".format(i), np.shape(data[i]))
    
        # preprocess similarity matrix for t-sne kl loss
        P[i] = P_preprocess(P[i])
        P[i] = torch.from_numpy(P[i]).float().to(device)

        
    net = IntegrAO(feature_dims, hidden_channels, embedding_dims, num_classes=num_classes).to(device)  # should load pre-trained model
    
    if model_path is not None:
        Project_GNN = _load_pre_trained_weights(net, model_path, device)
    else:
        Project_GNN = init_model(net, device, restore=None)
    Project_GNN.train()

    optimizer = torch.optim.Adam(Project_GNN.parameters(), lr=1e-1)
    c_mse = nn.MSELoss()
    c_cn = nn.CrossEntropyLoss()

    for epoch in range(alighment_epochs):
        adjust_learning_rate(optimizer, epoch)

        loss = 0
        embeddings = []

        kl_loss = np.array(0)
        kl_loss = torch.from_numpy(kl_loss).to(device).float()

        # KL loss for each network
        embeddings, _, pred, _ = Project_GNN(x_dict, edge_index_dict, dict_original_order)
        embeddings = list(embeddings.values())

        for i, X_embedding in enumerate(embeddings):
            kl_loss += tsne_loss(P[i], X_embedding)

        # pairwise alignment loss between each pair of networks
        alignment_loss = np.array(0)
        alignment_loss = torch.from_numpy(alignment_loss).to(device).float()

        for i in range(dataset_num - 1):
            for j in range(i + 1, dataset_num):
                low_dim_set1 = embeddings[i][dicts_commonIndex[(i, j)]]
                low_dim_set2 = embeddings[j][dicts_commonIndex[(j, i)]]
                alignment_loss += c_mse(low_dim_set1, low_dim_set2)

        loss += kl_loss + alignment_loss

        # if classification task, take the average of all the embeddings and calculate the classification loss
        clf_loss = c_cn(pred, labels)
        loss += clf_loss
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch) % 100 == 0:
            print(
                "epoch {}: loss {}, kl_loss:{:4f}, align_loss:{:4f}, clf_loss:{:4f}".format(
                    epoch, loss.data.item(), kl_loss.data.item(), alignment_loss.data.item(), clf_loss.data.item()
                )
            )
        # if epoch == 100:
        #     for i in range(dataset_num):
        #         P[i] = P[i] / 4.0

    # get the final embeddings for all samples
    embeddings, X_embedding_avg, preds, _ = Project_GNN(x_dict, edge_index_dict, dict_original_order)
    pred = pred.detach().cpu().numpy()
            
    # Now I need to put X_embedding_avg in order
    final_embeddings = X_embedding_avg.detach().cpu().numpy()

    end_time = time.time()
    print("Manifold alignment ends! Times: {}s".format(end_time - start_time))

    return final_embeddings, Project_GNN, preds
