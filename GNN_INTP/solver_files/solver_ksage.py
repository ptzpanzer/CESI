# implementation from https://github.com/tufts-ml/kcn-torch/tree/master

import json
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.nn import knn_graph
from torch_geometric.nn.models import GraphSAGE
import sklearn
import sklearn.neighbors

import support_functions


required_settings = {
    # run_mode
    "debug": None,
    
    # wandb settings
    "agent_id": None,

    # HPC settings
    "job_id": None,
    "coffer_slot": None,

    # Dataset settings
    "dataset": None,
    "fold": None,
    "holdout": None,

    # Model settings
    "model": "KSAGE",
    'k': 5,
    'conv_dim': 256,

    # Training settings
    'seed': 1,
    'full_batch': 64,
    'real_batch': 64,
    'epoch': 100,
    'nn_lr': 1e-3,
    'es_mindelta': 0.0,
    'es_endure': 3,
}


def extract_first_element_per_batch(tensor1, tensor2):
    # Get the unique batch indices from tensor2
    unique_batch_indices = torch.unique(tensor2)
    # Initialize a list to store the first elements of each batch item
    first_elements = []
    # Iterate through each unique batch index
    for batch_idx in unique_batch_indices:
        # Find the first occurrence of the batch index in tensor2
        first_occurrence = torch.nonzero(tensor2 == batch_idx, as_tuple=False)[0, 0]
        # Extract the first element from tensor1 and append it to the list
        first_element = tensor1[first_occurrence]
        first_elements.append(first_element)
    # Convert the list to a tensor
    result = torch.stack(first_elements, dim=0)
    return result


class INTP_Model(nn.Module):
    """
        GCN
    """
    def __init__(self, settings, device):
        super(INTP_Model, self).__init__()

        self.settings = settings
        self.device = device

        origin_path = f"./Datasets/{settings['dataset']}/"
        with open(origin_path + f'meta_data.json', 'r') as f:
            self.dataset_info = json.load(f)

        settings['num_features_in'] = len(self.dataset_info["op_dic"]) + len(list(self.dataset_info["non_eu_col"].keys()))
        settings['num_features_out'] = 1

        num_features_in = settings["num_features_in"]
        num_features_out = settings["num_features_out"]
        k = settings["k"]
        conv_dim = settings["conv_dim"]

        self.device = device
        self.num_features_in = num_features_in
        self.k = k
        self.length_scale = 0.5
        
        self.gsage = GraphSAGE(in_channels=num_features_in, hidden_channels=settings['conv_dim'], num_layers=2, out_channels=num_features_out).to(self.device)
        
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=5e-4, lr=settings['nn_lr'])
        self.scheduler = None

    def forward(self, batch):
        inputs, coords, targets, input_lenths = batch
        head_only = True

        start_index = 0
        all_x_list = []
        all_y_list = []
        all_i_list = []
        all_ei_list = []
        all_ew_list = []
        for i in range(inputs.size(0)):
            inp_l = input_lenths[i]
            inp, coo, tar = inputs[i, :inp_l, :], coords[i, :inp_l, :], targets[i, :inp_l, :]

            # print(f"inp: {inp.size()}")
            # print(f"coo: {coo.size()}")
            # print(f"tar: {tar.size()}")

            if self.k <= len(coo[1:, :]):
                k = self.k
            else:
                k = len(coo[1:, :])
            knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k).fit(coo[1:, :].cpu())
            neighbors = knn.kneighbors(coo[0:1, :].cpu(), return_distance=False)

            # print(f"neighbors: {neighbors}")
            # print(f"inp[0:1, :]: {inp[0:1, :].size()}")
            # print(f"inp[neighbors, :]: {inp[neighbors, :].size()}")

            inp_new = torch.concat([inp[0:1, :], inp[neighbors, :].squeeze(0)], axis=0)
            all_x_list.append(inp_new)
            tar_new = torch.concat([tar[0:1, :], tar[neighbors, :].squeeze(0)], axis=0)
            all_y_list.append(tar_new)
            indexer = torch.full((tar_new.size(0), ), i)
            all_i_list.append(indexer)

            all_coords = torch.concat([coo[0:1, :], coo[neighbors, :].squeeze(0)], axis=0)
            kernel = sklearn.metrics.pairwise.rbf_kernel(all_coords.cpu().numpy(), gamma=1 / (2 * self.length_scale ** 2))
            adj = torch.from_numpy(kernel).to(self.device)

            nz = adj.nonzero(as_tuple=True)
            edge_index = torch.stack(nz, dim=0) + start_index
            all_ei_list.append(edge_index)
            edge_weight = adj[nz]
            all_ew_list.append(edge_weight)

            start_index += tar_new.size(0)

        x_l = torch.cat(all_x_list, dim=0)
        y_l = torch.cat(all_y_list, dim=0)
        indexer = torch.cat(all_i_list, dim=0)
        edge_index = torch.cat(all_ei_list, dim=1)
        edge_weight = torch.cat(all_ew_list, dim=0)

        # print(f"x_l: {x_l.size()}")
        # print(f"y_l: {y_l.size()}")
        # print(f"indexer: {indexer.size()}")
        # print(f"edge_index: {edge_index.size()}")
        # print(f"edge_weight: {edge_weight.size()}")

        output = self.gsage(x_l, edge_index, edge_weight)
        
        if not head_only:
            return output, y_l
        else:
            output_head = extract_first_element_per_batch(output, indexer)
            target_head = extract_first_element_per_batch(y_l, indexer)
            return output_head, target_head

    def loss_func(self, model_output, target):
        loss = torch.nn.L1Loss()(model_output, target)
        return loss, model_output, target
