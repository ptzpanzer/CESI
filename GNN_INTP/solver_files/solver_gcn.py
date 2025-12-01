import json
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, knn_graph

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
    "model": "GCN",
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


# Helper function for 2+d distance
def newDistance(a, b):
    return torch.norm(a - b, dim=-1)


# Helper function for edge weights
def makeEdgeWeight(x, edge_index):
    to = edge_index[0]
    fro = edge_index[1]
    
    distances = newDistance(x[to], x[fro])
    max_val = torch.max(distances)
    min_val = torch.min(distances)
    
    rng = max_val - min_val
    edge_weight = (max_val - distances) / rng
    
    return edge_weight


def padded_seq_to_vectors(padded_seq, logger):
    # Get the actual lengths of each sequence in the batch
    actual_lengths = logger.int()
    # Step 1: Form the first tensor containing all actual elements from the batch
    mask = torch.arange(padded_seq.size(1), device=padded_seq.device) < actual_lengths.view(-1, 1)
    tensor1 = torch.masked_select(padded_seq, mask.unsqueeze(-1)).view(-1, padded_seq.size(-1))
    # Step 2: Form the second tensor to record which row each element comes from
    tensor2 = torch.repeat_interleave(torch.arange(padded_seq.size(0), device=padded_seq.device), actual_lengths)
    return tensor1, tensor2


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
        self.conv1 = GCNConv(num_features_in, conv_dim).to(self.device)
        self.conv2 = GCNConv(conv_dim, conv_dim).to(self.device)
        self.fc = nn.Linear(conv_dim, num_features_out).to(self.device)
        
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=settings['nn_lr'])
        self.scheduler = None

    def forward(self, batch):
        inputs, coords, targets, input_lenths = batch
        head_only = True

        x_l, indexer = padded_seq_to_vectors(inputs, input_lenths)
        y_l, _ = padded_seq_to_vectors(targets, input_lenths)
        c_l, _ = padded_seq_to_vectors(coords, input_lenths)
        
        edge_index = knn_graph(c_l, k=self.k, batch=indexer)
        edge_weight = makeEdgeWeight(c_l, edge_index).to(self.device)

        h1 = F.relu(self.conv1(x_l, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)
        
        if not head_only:
            return output, y_l
        else:
            output_head = extract_first_element_per_batch(output, indexer)
            target_head = extract_first_element_per_batch(y_l, indexer)
            return output_head, target_head

    def loss_func(self, model_output, target):
        loss = torch.nn.L1Loss()(model_output, target)
        return loss, model_output, target
