from integrao.unsupervised_train import tsne_p_deep
from integrao.supervised_train import tsne_p_deep_classification

from integrao.main import dist2, integrao_fuse, _stable_normalized
from integrao.util import data_indexing

import snf
import pandas as pd
import numpy as np
import os

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from integrao.dataset import GraphDataset


class integrao_integrater(object):
    def __init__(
        self,
        datasets,
        dataset_name=None,
        modalities_name_list=None,
        neighbor_size=None,
        embedding_dims=50,
        fusing_iteration=20,
        normalization_factor=1.0,
        alighment_epochs=1000,
        beta=1.0,
        mu=0.5,
        random_state=42,
    ):
        self.datasets = datasets
        self.dataset_name = dataset_name
        self.modalities_name_list = modalities_name_list
        self.embedding_dims = embedding_dims
        self.fusing_iteration = fusing_iteration
        self.normalization_factor = normalization_factor
        self.alighment_epochs = alighment_epochs
        self.beta = beta
        self.mu = mu
        self.random_state=random_state

        # data indexing
        (
            self.dicts_common,
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            self.dicts_unique,
            self.original_order,
            self.dict_original_order,
        ) = data_indexing(self.datasets)

        # set neighbor size
        if neighbor_size == None:
            self.neighbor_size = int(datasets[0].shape[0] / 6)
        else:
            self.neighbor_size = neighbor_size
        print("Neighbor size:", self.neighbor_size)

    def network_diffusion(self):
        S_dfs = []
        for i in range(0, len(self.datasets)):
            view = self.datasets[i]
            dist_mat = dist2(view.values, view.values)
            S_mat = snf.compute.affinity_matrix(
                dist_mat, K=self.neighbor_size, mu=self.mu
            )

            S_df = pd.DataFrame(
                data=S_mat, index=self.original_order[i], columns=self.original_order[i]
            )

            S_dfs.append(S_df)

        self.fused_networks = integrao_fuse(
            S_dfs.copy(),
            dicts_common=self.dicts_common,
            dicts_unique=self.dicts_unique,
            original_order=self.original_order,
            neighbor_size=self.neighbor_size,
            fusing_iteration=self.fusing_iteration,
            normalization_factor=self.normalization_factor,
        )
        return self.fused_networks

    def unsupervised_alignment(self):
        # turn pandas dataframe into np array
        datasets_val = [x.values for x in self.datasets]
        fused_networks_val = [x.values for x in self.fused_networks]

        S_final, self.models = tsne_p_deep(
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            datasets_val,
            P=fused_networks_val,
            neighbor_size=self.neighbor_size,
            embedding_dims=self.embedding_dims,
            alighment_epochs=self.alighment_epochs,
        )

        self.final_embeds = pd.DataFrame(
            data=S_final, index=self.dict_sampleToIndexs.keys()
        )
        self.final_embeds.sort_index(inplace=True)

        # calculate the final similarity graph
        dist_final = dist2(self.final_embeds.values, self.final_embeds.values)
        Wall_final = snf.compute.affinity_matrix(
            dist_final, K=self.neighbor_size, mu=self.mu
        )

        Wall_final = _stable_normalized(Wall_final)

        return self.final_embeds, Wall_final, self.models

    def classification_finetuning(self, clf_labels, model_path, finetune_epochs=1000):
        # turn pandas dataframe into np array
        datasets_val = [x.values for x in self.datasets]
        fused_networks_val = [x.values for x in self.fused_networks]

        # reorder of clf_labels to make it the same with self.dict_sampleToIndexs.keys()
        clf_labels = clf_labels.loc[self.dict_sampleToIndexs.keys()]

        S_final, self.models, preds = tsne_p_deep_classification(
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            self.dict_original_order,
            datasets_val,
            clf_labels,
            P=fused_networks_val,
            model_path=model_path,
            neighbor_size=self.neighbor_size,
            embedding_dims=self.embedding_dims,
            alighment_epochs=finetune_epochs,
            num_classes=len(np.unique(clf_labels)),
        )

        self.final_embeds = pd.DataFrame(
            data=S_final, index=self.dict_sampleToIndexs.keys()
        )
        self.final_embeds.sort_index(inplace=True)

        # calculate the final similarity graph
        dist_final = dist2(self.final_embeds.values, self.final_embeds.values)
        Wall_final = snf.compute.affinity_matrix(
            dist_final, K=self.neighbor_size, mu=self.mu
        )

        Wall_final = _stable_normalized(Wall_final)

        return self.final_embeds, Wall_final, self.models, preds


class integrao_predictor(object):
    def __init__(
        self,
        datasets,
        dataset_name=None,
        modalities_name_list=None,
        neighbor_size=None,
        embedding_dims=50,
        hidden_channels=128,
        fusing_iteration=20,
        normalization_factor=1.0,
        alighment_epochs=1000,
        beta=1.0,
        mu=0.5,
        num_classes=None,       
    ):
        self.datasets = datasets
        self.dataset_name = dataset_name
        self.modalities_name_list = modalities_name_list
        self.embedding_dims = embedding_dims
        self.hidden_channels = hidden_channels
        self.fusing_iteration = fusing_iteration
        self.normalization_factor = normalization_factor
        self.alighment_epochs = alighment_epochs
        self.beta = beta
        self.mu = mu
        self.num_classes = num_classes

        # data indexing
        (
            self.dicts_common,
            self.dicts_commonIndex,
            self.dict_sampleToIndexs,
            self.dicts_unique,
            self.original_order,
            self.dict_original_order,
        ) = data_indexing(self.datasets)

        # set neighbor size
        if neighbor_size == None:
            self.neighbor_size = int(datasets[0].shape[0] / 6)
        else:
            self.neighbor_size = neighbor_size
        print("Neighbor size:", self.neighbor_size)

        self.feature_dims = []
        for i in range(len(self.datasets)):
            self.feature_dims.append(np.shape(self.datasets[i])[1])

        if num_classes is not None:
            self.num_classes = num_classes
        

    def network_diffusion(self):
        S_dfs = []
        for i in range(0, len(self.datasets)):
            view = self.datasets[i]
            dist_mat = dist2(view.values, view.values)
            S_mat = snf.compute.affinity_matrix(
                dist_mat, K=self.neighbor_size, mu=self.mu
            )

            S_df = pd.DataFrame(
                data=S_mat, index=self.original_order[i], columns=self.original_order[i]
            )

            S_dfs.append(S_df)

        self.fused_networks = integrao_fuse(
            S_dfs.copy(),
            dicts_common=self.dicts_common,
            dicts_unique=self.dicts_unique,
            original_order=self.original_order,
            neighbor_size=self.neighbor_size,
            fusing_iteration=self.fusing_iteration,
            normalization_factor=self.normalization_factor,
        )
        return self.fused_networks


    def _load_pre_trained_weights(self, model, model_path, device):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model
    
    def inference_unsupervised(self, model_path, new_datasets, modalities_names):
        # loop through the new_dataset and create Graphdatase
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from integrao.IntegrAO_unsupervised import IntegrAO
        model = IntegrAO(self.feature_dims, self.hidden_channels, self.embedding_dims).to(device)
        model = self._load_pre_trained_weights(model, model_path, device)

        x_dict = {}
        edge_index_dict = {}
        for i, modal in enumerate(new_datasets):
            # find the index of the modal in the self.modalities_name_list
            model_name = modalities_names[i]
            modal_index = self.modalities_name_list.index(model_name)

            dataset = GraphDataset(
                self.neighbor_size,
                modal.values,
                self.fused_networks[modal_index].values,
                transform=T.ToDevice(device),
            )
            modal_dg = dataset[0]

            x_dict[modal_index] = modal_dg.x
            edge_index_dict[modal_index] = modal_dg.edge_index

        # Now to do the inference
        # ---------------------------------------------------------
        embeddings= model(x_dict, edge_index_dict)
        for i in range(len(new_datasets)):
            embeddings[i] = embeddings[i].detach().cpu().numpy()  

        final_embedding = np.array([]).reshape(0, self.embedding_dims)
        for key in self.dict_sampleToIndexs:
            sample_embedding = np.zeros((1, self.embedding_dims))

            for (dataset, index) in self.dict_sampleToIndexs[key]:
                sample_embedding += embeddings[dataset][index]
            sample_embedding /= len(self.dict_sampleToIndexs[key])

            final_embedding = np.concatenate((final_embedding, sample_embedding), axis=0)

        # Now format the final embeddings
        # ---------------------------------------------------------
        final_embedding_df = pd.DataFrame(
            data=final_embedding, index=self.dict_sampleToIndexs.keys()
        )
        final_embedding_df.sort_index(inplace=True)

        # calculate the final similarity graph
        dist_final = dist2(final_embedding_df.values, final_embedding_df.values)
        Wall_final = snf.compute.affinity_matrix(
            dist_final, K=self.neighbor_size, mu=self.mu
        )

        Wall_final = _stable_normalized(Wall_final)

        return final_embedding_df, Wall_final
    
    def interpret_unsupervised(self, model_path, result_dir, new_datasets, modalities_names):
        # loop through the new_dataset and create Graphdatase
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from integrao.IntegrAO_unsupervised import IntegrAO
        model = IntegrAO(self.feature_dims, self.hidden_channels, self.embedding_dims).to(device)
        model = self._load_pre_trained_weights(model, model_path, device)


        # explain the model
        from captum.attr import IntegratedGradients

        # It takes as input the variable node features for one domain,
        # while the remaining features and edge indices remain fixed.
        def custom_forward(x, static_x_dict, edge_index_dict, domain):
            x_dict = static_x_dict.copy()
            x_dict[domain] = x 

            out_dict = model(x_dict, edge_index_dict)

            return out_dict[domain].sum(dim=1)   # iG requires scalar output; so we sum the output of the embeddings

        # prepare the data
        x_dict = {}
        edge_index_dict = {}
        for i, modal in enumerate(new_datasets):
            model_name = modalities_names[i]
            modal_index = self.modalities_name_list.index(model_name)

            dataset = GraphDataset(
                self.neighbor_size,
                modal.values,
                self.fused_networks[modal_index].values,
                transform=T.ToDevice(device),
            )
            modal_dg = dataset[0]

            x_dict[modal_index] = modal_dg.x
            edge_index_dict[modal_index] = modal_dg.edge_index

        # Loop over each domain (modality)
        # ---------------------------------------------------------
        feat_importances = {}
        for domain in x_dict:
            x_input = x_dict[domain] # The variable input for the current domain.
            static_x = {k: x_dict[k] for k in x_dict}

            ig = IntegratedGradients(custom_forward)

            attributions, delta = ig.attribute(
                inputs=x_input,
                additional_forward_args=(static_x, edge_index_dict, domain),
                return_convergence_delta=True
            )

            if domain not in feat_importances:
                feat_importances[domain] = []
            feat_importances[domain].append(attributions.detach().cpu().numpy())


        df_list = []
        for domain in feat_importances:

            # Concatenate along the first axis (nodes).
            feat_importances[domain] = np.concatenate(feat_importances[domain], axis=0)
            num_feats = feat_importances[domain].shape[1]
            # Create a DataFrame; here columns are named feat_0, feat_1, etc.
            df = pd.DataFrame(feat_importances[domain], columns=[f'feat_{i}' for i in range(num_feats)])
            df_list.append(df)

            # save the feature importance
            csv_path = os.path.join(result_dir, f'{modalities_names[domain]}_feat_importance.csv')
            df.to_csv(csv_path, index=False)
            print(df.shape)

            print(f"Saved feature importances for domain {modalities_names[domain]} to {csv_path}")

        return  df_list


    def inference_supervised(self, model_path, new_datasets, modalities_names):
        # loop through the new_dataset and create Graphdatase
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from integrao.IntegrAO_supervised import IntegrAO
        model = IntegrAO(self.feature_dims, self.hidden_channels, self.embedding_dims, num_classes=self.num_classes).to(device)
        model = self._load_pre_trained_weights(model, model_path, device)

        x_dict = {}
        edge_index_dict = {}
        for i, modal in enumerate(new_datasets):
            # find the index of the modal in the self.modalities_name_list
            model_name = modalities_names[i]
            modal_index = self.modalities_name_list.index(model_name)

            dataset = GraphDataset(
                self.neighbor_size,
                modal.values,
                self.fused_networks[modal_index].values,
                transform=T.ToDevice(device),
            )
            modal_dg = dataset[0]

            x_dict[modal_index] = modal_dg.x
            edge_index_dict[modal_index] = modal_dg.edge_index

        # Now to do the inference
        final_embeddings, _, preds, id_list = model(
            x_dict, edge_index_dict, self.dict_original_order
        )

        preds = F.softmax(preds, dim=1)
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)

        return preds


    def interpret_supervised(self, model_path, result_dir, new_datasets, modalities_names):
        # loop through the new_dataset and create Graphdatase
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        from integrao.IntegrAO_supervised import IntegrAO
        model = IntegrAO(self.feature_dims, self.hidden_channels, self.embedding_dims, num_classes=self.num_classes).to(device)
        model = self._load_pre_trained_weights(model, model_path, device)

        # explain the model
        from captum.attr import IntegratedGradients

        # It takes variable node features (x) for a given domain,
        # while keeping the rest of the inputs (static_x_dict, edge_index_dict, and domain_sample_ids) fixed.
        def custom_forward(x, static_x_dict, edge_index_dict, domain, domain_sample_ids):
            x_dict = static_x_dict.copy()
            x_dict[domain] = x 

            _, _, output, _ = model(x_dict, edge_index_dict, domain_sample_ids)

            # Aggregate output per sample to a scalar.
            # Here we sum over the class dimension (dim=1); adjust if you need a different reduction; for example just a single class.
            return output.sum(dim=1)


        # Prepare the data dictionaries for node features and edge indices.
        x_dict = {}
        edge_index_dict = {}
        for i, modal in enumerate(new_datasets):
            # find the index of the modal in the self.modalities_name_list
            model_name = modalities_names[i]
            modal_index = self.modalities_name_list.index(model_name)

            dataset = GraphDataset(
                self.neighbor_size,
                modal.values,
                self.fused_networks[modal_index].values,
                transform=T.ToDevice(device),
            )
            modal_dg = dataset[0]

            x_dict[modal_index] = modal_dg.x
            edge_index_dict[modal_index] = modal_dg.edge_index

        # Compute feature importances using IntegratedGradients.
        feat_importances = {}
        for domain in x_dict:
            x_input = x_dict[domain]
            static_x = {k: x_dict[k] for k in x_dict}

            ig = IntegratedGradients(custom_forward)

            attributions, delta = ig.attribute(
                inputs=x_input,
                additional_forward_args=(static_x, edge_index_dict, domain, self.dict_original_order),
                return_convergence_delta=True
            )

            if domain not in feat_importances:
                feat_importances[domain] = []
            feat_importances[domain].append(attributions.detach().cpu().numpy())


        df_list = []
        for domain in feat_importances:

            # Concatenate along the first axis (nodes).
            feat_importances[domain] = np.concatenate(feat_importances[domain], axis=0)
            num_feats = feat_importances[domain].shape[1]
            # Create a DataFrame; here columns are named feat_0, feat_1, etc.
            df = pd.DataFrame(feat_importances[domain], columns=[f'feat_{i}' for i in range(num_feats)])
            df_list.append(df)

            # save the feature importance
            csv_path = os.path.join(result_dir, f'{modalities_names[domain]}_feat_importance.csv')
            df.to_csv(csv_path, index=False)
            print(df.shape)

            print(f"Saved feature importances for domain {modalities_names[domain]} to {csv_path}")
        
        return df_list