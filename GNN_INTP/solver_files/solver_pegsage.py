# implementation from https://github.com/konstantinklemmer/pe-gnn

import json
import os
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import knn_graph
from torch_geometric.nn.models import GraphSAGE

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
    "model": "PEGSAGE",
    'emb_hidden_dim': 256,
    'emb_dim': 64,
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


class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]
        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                 output_dim,
                 dropout_rate=None,
                 activation="sigmoid",
                 use_layernormalize=False,
                 skip_connection=False,
                 context_str=''):
        '''
        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output


class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                 output_dim,
                 num_hidden_layers=0,
                 dropout_rate=0.5,
                 hidden_dim=-1,
                 activation="relu",
                 use_layernormalize=True,
                 skip_connection=False,
                 context_str=None):
        '''
        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))
        else:
            self.layers.append(SingleFeedForwardNN(input_dim=self.input_dim,
                                                   output_dim=self.hidden_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=self.use_layernormalize,
                                                   skip_connection=self.skip_connection,
                                                   context_str=self.context_str))

            for i in range(self.num_hidden_layers - 1):
                self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                       output_dim=self.hidden_dim,
                                                       dropout_rate=self.dropout_rate,
                                                       activation=self.activation,
                                                       use_layernormalize=self.use_layernormalize,
                                                       skip_connection=self.skip_connection,
                                                       context_str=self.context_str))

            self.layers.append(SingleFeedForwardNN(input_dim=self.hidden_dim,
                                                   output_dim=self.output_dim,
                                                   dropout_rate=self.dropout_rate,
                                                   activation=self.activation,
                                                   use_layernormalize=False,
                                                   skip_connection=False,
                                                   context_str=self.context_str))

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        
        output = input_tensor
        for layer in self.layers:
            output = layer(output)

        return output


def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    freq_list = None
    if freq_init == "random":
        freq_list = torch.rand(frequency_num) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0 - 1))
        timescales = min_radius * torch.exp(torch.arange(frequency_num, dtype=torch.float32) * log_timescale_increment)
        freq_list = 1.0 / timescales
    return freq_list


class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
                 max_radius=0.01, min_radius=0.00001,
                 freq_init="geometric",
                 ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

        if self.ffn is not None:
            self.ffn = MultiLayerFeedForwardNN(2 * frequency_num * 2, spa_embed_dim)

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = torch.unsqueeze(self.freq_list, 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = freq_mat.repeat(1, 2)

    def make_input_embeds(self, coords):
        # coords: shape (batch_size, num_context_pt, 2)
        batch_size, num_context_pt, _ = coords.shape
        # coords: shape (batch_size, num_context_pt, 2, 1, 1)
        coords = coords.unsqueeze(-1).unsqueeze(-1)
        # coords: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords = coords.repeat(1, 1, 1, self.frequency_num, 2)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords * self.freq_mat.to(self.device)
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = torch.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = torch.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1
        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = torch.reshape(spr_embeds, (batch_size, num_context_pt, -1))
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


class INTP_Model(nn.Module):
    """
        GAT with positional encoder and auxiliary tasks
    """

    def __init__(self, settings, device):
        super(INTP_Model, self).__init__()

        self.settings = settings
        self.device = device

        origin_path = f"./Datasets/{settings['dataset']}/"
        with open(origin_path + f'meta_data.json', 'r') as f:
            self.dataset_info = json.load(f)

        if len(list(self.dataset_info["eu_col"].keys())) == 2:
            settings['num_features_in'] = len(self.dataset_info["op_dic"]) + len(list(self.dataset_info["non_eu_col"].keys()))
        if len(list(self.dataset_info["eu_col"].keys())) > 2:
            settings['num_features_in'] = len(self.dataset_info["op_dic"]) + len(list(self.dataset_info["non_eu_col"].keys())) + len(list(self.dataset_info["eu_col"].keys())) - 2
        settings['num_features_out'] = 1

        num_features_in = settings["num_features_in"]
        num_features_out = settings["num_features_out"]
        emb_hidden_dim = settings["emb_hidden_dim"]
        emb_dim = settings["emb_dim"]
        k = settings["k"]
        conv_dim = settings["conv_dim"]

        self.device = device
        self.num_features_in = num_features_in
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
        self.k = k
        
        self.spenc = GridCellSpatialRelationEncoder(
            spa_embed_dim=emb_hidden_dim, ffn=True, min_radius=1e-06, max_radius=360
        ).to(self.device)
        self.dec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 4, emb_dim)
        ).to(self.device)
        self.gsage = GraphSAGE(in_channels=num_features_in + emb_dim, hidden_channels=conv_dim, num_layers=2, out_channels=num_features_out).to(self.device)
        
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=settings['nn_lr'])
        self.scheduler = None

    def forward(self, batch):
        inputs, coords, targets, input_lenths = batch
        head_only = True

        if coords.size(2) == 2:
            emb = self.spenc(coords)
        elif coords.size(2) > 2:
            emb = self.spenc(coords[:, :, 0:2])
        emb = self.dec(emb)
        
        emb_l, indexer = padded_seq_to_vectors(emb, input_lenths)
        x_l, _ = padded_seq_to_vectors(inputs, input_lenths)
        y_l, _ = padded_seq_to_vectors(targets, input_lenths)
        c_l, _ = padded_seq_to_vectors(coords, input_lenths)
        
        edge_index = knn_graph(c_l, k=self.k, batch=indexer)
        edge_weight = makeEdgeWeight(c_l, edge_index).to(self.device)

        if coords.size(2) == 2:
            x = torch.cat((x_l, emb_l), dim=1)
        elif coords.size(2) > 2:
            x = torch.cat((x_l, emb_l, c_l[:, 2:]), dim=1)
        output = self.gsage(x, edge_index, edge_weight)
        
        if not head_only:
            return output, y_l
        else:
            output_head = extract_first_element_per_batch(output, indexer)
            target_head = extract_first_element_per_batch(y_l, indexer)
            return output_head, target_head
        
    def loss_func(self, model_output, target):
        loss = torch.nn.L1Loss()(model_output, target)
        return loss, model_output, target
        