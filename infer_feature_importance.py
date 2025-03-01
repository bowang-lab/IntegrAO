import numpy as np
import pandas as pd
import snf
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt

import sys
import os
import argparse
import torch

import umap
from sklearn.model_selection import train_test_split


# Add the parent directory of "integrao" to the Python path
module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from integrao.integrater import integrao_predictor


dataset_name = 'cancer_omics_prediction'

# create result dir
result_dir = os.path.join(
    module_path, "results/{}".format(dataset_name)
)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Hyperparameters
neighbor_size = 20
embedding_dims = 64
fusing_iteration = 30
normalization_factor = 1.0
alighment_epochs = 1000
beta = 1.0
mu = 0.5

dataset_name = 'cancer omics'
cluster_number = 15

testdata_dir = os.path.join(module_path, "data/omics/")

methyl_ = os.path.join(testdata_dir, "omics1.txt")
expr_ = os.path.join(testdata_dir, "omics2.txt")
protein_ = os.path.join(testdata_dir, "omics3.txt")
truelabel = os.path.join(testdata_dir, "clusters.txt")


methyl = pd.read_csv(methyl_, index_col=0, delimiter="\t")
expr = pd.read_csv(expr_, index_col=0, delimiter="\t")
protein = pd.read_csv(protein_, index_col=0, delimiter="\t")
truelabel = pd.read_csv(truelabel, index_col=0, delimiter="\t")

methyl = np.transpose(methyl)
expr = np.transpose(expr)
protein = np.transpose(protein)
print(methyl.shape)
print(expr.shape)
print(protein.shape)
print(truelabel.shape)
print("finish loading data!")


# Network fusion for the whole graph
predictor = integrao_predictor(
    [methyl, expr, protein],
    dataset_name,
    modalities_name_list=["methyl", "expr", "protein"], 
    neighbor_size=neighbor_size,
    embedding_dims=embedding_dims,
    fusing_iteration=fusing_iteration,
    normalization_factor=normalization_factor,
    alighment_epochs=alighment_epochs,
    beta=beta,
    mu=mu,
)
# data indexing
fused_networks = predictor.network_diffusion()

model_path = os.path.join(result_dir, "model_integrao_unsupervised.pth")
preds = predictor.inference_unsupervised(model_path, new_datasets=[methyl, expr, protein], modalities_names=["methyl", "expr", "protein"])

import ipdb; ipdb.set_trace()
assert 0

labels = spectral_clustering(S_final, n_clusters=cluster_number)

true_labels = truelabel.sort_values('subjects')['cluster.id'].tolist()

score_all = v_measure_score(true_labels, labels)
print("IntegrAO for clustering union 500 samples NMI score: ", score_all)





