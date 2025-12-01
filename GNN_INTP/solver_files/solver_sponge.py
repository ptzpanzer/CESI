import json
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, knn_graph
from sklearn.neighbors import NearestNeighbors
import scipy

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
    "model": "SPONGE",
    'k': 5,
    'hidden_sizes_0': 5,
    'hidden_sizes_1': 10,
    'length_scale': [1, 0.1, 0.01, 0.001],

    # Training settings
    'seed': 1,
    'full_batch': 64,
    'real_batch': 64,
    'epoch': 100,
    'nn_lr': 1e-2,
    'es_mindelta': 0.0,
    'es_endure': 3,
}


class MyConstraints:
    def __call__(self, w):
        # 保证参数非负
        w = w * (w >= 0).float()
        return w


# Build a graph convolutional layer
class MyAdjconvLayer(nn.Module):
    def __init__(self, settings):
        super(MyAdjconvLayer, self).__init__()
        self.constraint = MyConstraints()
        self.adjconv = nn.Parameter(torch.randn(len(settings['length_scale']), 1))

    def forward(self, Adj_t, H_t):
        self.adjconv.data = self.constraint(self.adjconv.data)
        
        adjconv = self.adjconv / torch.sum(self.adjconv)

        batch, _, n1, n2 = Adj_t.shape
        D1 = torch.tensordot(Adj_t, adjconv, dims=([1], [0])).view(batch, n1, n2)
        D2 = torch.sum(D1, dim=2).view(-1, n1, 1).float()
        output = D1 / D2

        # print(output.size(), H_t.size())

        return output, H_t


# Build a graph convolutional layer
class MyGraphconvNeigh(nn.Module):
    def __init__(self, in_size, hidden_out, k):
        super(MyGraphconvNeigh, self).__init__()
        self.in_size = in_size
        self.num_outputs = hidden_out
        self.num_neigh = k
        self.graphconvNeig = nn.Parameter(torch.randn(self.in_size, self.num_outputs))

    def forward(self, Adj, H):

        result0 = torch.matmul(Adj, H)
        output = torch.matmul(result0, self.graphconvNeig)
        Adj_new = Adj[:, 0, 0:self.num_neigh + 1].view(Adj.shape[0], 1, self.num_neigh + 1)

        # print(Adj_new.size(), output.size())
        
        return Adj_new, output


# Build a graph convolutional layer
class MyGraphconvLayer(nn.Module):
    def __init__(self, in_size, hidden_out):
        super(MyGraphconvLayer, self).__init__()
        self.in_size = in_size
        self.num_outputs = hidden_out
        self.graphconv = nn.Parameter(torch.randn(self.in_size, self.num_outputs))

    def forward(self, A, x):
        result0 = torch.matmul(A, x)
        output = torch.matmul(result0, self.graphconv)
        return output


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

        in_sizes = len(self.dataset_info["op_dic"]) + len(self.dataset_info["non_eu_col"]) + len(self.dataset_info["eu_col"]) + 2
        hidden_sizes_0 = self.settings['hidden_sizes_0']
        hidden_sizes_1 = self.settings['hidden_sizes_1']
        output_dim = 1

        self.hidden0 = MyAdjconvLayer(self.settings)
        self.hidden1 = MyGraphconvNeigh(in_sizes, hidden_sizes_0, self.settings['k'])
        self.hidden2 = MyGraphconvLayer(hidden_sizes_0, hidden_sizes_1)
        self.dense = nn.Linear(hidden_sizes_1, output_dim, bias=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=settings['nn_lr'])
        self.scheduler = None

    def forward(self, batch):
        Adj_t, H_t, Y_t = batch
        
        output, H_t = self.hidden0(Adj_t, H_t)
        A, x = self.hidden1(output, H_t)
        x = F.relu(x)
        x = self.hidden2(A, x)
        x = F.relu(x)
        x = self.dense(x)
        x = x[:, 0, :]

        y = Y_t[:, 0, :]

        # print(x.size(), y.size())

        return x, y

    def loss_func(self, model_output, target):
        loss = torch.nn.MSELoss()(model_output, target)
        return loss, model_output, target
