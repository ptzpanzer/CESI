import json
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
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
    "model": "SMACNP",
    'num_hidden': 128,
    
    # Training settings
    'seed': 1,
    'full_batch': 64,
    'real_batch': 64,
    'epoch': 100,
    'nn_lr': 1e-3,
    'es_mindelta': 0.0,
    'es_endure': 3,
}


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)  # (bs,nc,128)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        # b, n, size = x.size()
        # x = x.view(b * n, size)
        return self.linear_layer(x)


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):  # 4
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k  # 4
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        attn = torch.softmax(attn, dim=-1) #(4*bs,nt,nc)

        # Dropout
        # attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)
        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden  # 128
        self.num_hidden_per_attn = num_hidden // h  # 128/4=32
        self.h = h  # 4

        self.key = Linear(num_hidden, num_hidden, bias=False)  # (bs,nc,128)
        self.value = Linear(num_hidden, num_hidden, bias=False)  # (bs,nc,128)
        self.query = Linear(num_hidden, num_hidden, bias=False)  # (bs,nc,128)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value,
                query):  # query: target_x(bs*nt,x_size), key: context_x(bs,nc,x_size), value: representation(bs,  ,12)
        batch_size = key.size(0)  # bs
        seq_k = key.size(1)  # nc
        seq_q = query.size(1)  # nt
        residual = query  # (bs,nt,x_size)

        # Make multihead
        # self.key((bs,nc,x_size))=(bs*nc,128)
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # (bs,nc,4,32)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)  # (bs,nc,4,32)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)  # (bs,nt,4,32)

        key = key.permute(2, 0, 1, 3).reshape(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).reshape(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).reshape(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.reshape(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).reshape(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1) #(bs,nt,128*2)

        # Final linear
        result = self.final_linear(result)  #(bs,nt,128*2)-->(bs,nt,128)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class LocationEncoder(nn.Module):
    """
     Mean-Location Encoder [w]
    """

    def __init__(self, x_size, y_size, c_size, num_hidden):
        super(LocationEncoder, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.c_size = c_size
        
        self.input_projection = Linear(self.c_size + self.y_size, int(num_hidden))     #s+y
        self.context_projection = Linear(self.c_size, int(num_hidden))    #s
        self.target_projection = Linear(int(num_hidden), num_hidden)
    def forward(self, context_x, context_y, target_x):
        encoder_input = torch.cat([context_x[..., 0:self.c_size], context_y], dim=-1)  # concat context location (x), context value (y)
        encoder_input = self.input_projection(encoder_input)  # (bs,nc,3)--> (bs,nc,num_hidden)
        value = encoder_input  # (bs,nc,num_hidden)
        key = self.context_projection(context_x[..., 0:self.c_size])  # (bs,nc,num_hidden)
        query = self.context_projection(target_x[..., 0:self.c_size])  # (bs,nt,num_hidden)
        query = torch.unsqueeze(query, axis=2)  # (bs,nt,num_hidden)-->(bs,nt,1,num_hidden)
        key = torch.unsqueeze(key, axis=1)  # (bs,nc,2)-->(bs,1,nc,num_hidden)
        weights = - torch.abs((key - query) * 0.5)  # [bs,nt,nc,num_hidden]
        weights = torch.sum(weights, axis=-1)  # [bs,nt,nc]
        weight = torch.softmax(weights, dim=-1)  # [bs,nt,nc]
        rep = torch.matmul(weight, value)  # (bs,nt,nc)*(bs,nc,hidden_number)=(bs,nt,hidden_number)
        rep = self.target_projection(rep)  # (bs,nt,hidden_number)

        return rep


class DeterministicEncoder(nn.Module):
    """
      Mean-attribute Encoder [r]
    """

    def __init__(self, x_size, y_size, c_size, num_hidden):
        super(DeterministicEncoder, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.c_size = c_size

        input_dim = self.x_size + self.y_size
        
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(input_dim - self.y_size, num_hidden)
        self.target_projection = Linear(input_dim - self.y_size, num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = torch.cat([context_x[..., self.c_size:], context_y], dim=-1)  # concat context location (x), context value (y)
        encoder_input = self.input_projection(encoder_input)
        query = self.target_projection(target_x[..., self.c_size:])
        keys = self.context_projection(context_x[..., self.c_size:])

        for attention in self.cross_attentions:  # cross attention layer
            query, _ = attention(keys, encoder_input, query)  # (bs,nt,hidden_number)

        return query


class VarianceEncoder(nn.Module):
    """
    Variance Encoder [v]
    """
    def __init__(self, x_size, y_size, c_size, num_hidden):
        super(VarianceEncoder, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.c_size = c_size

        input_dim = self.x_size
        
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim + self.c_size, num_hidden)
        self.context_projection = Linear(input_dim + self.c_size, num_hidden)
        self.target_projection = Linear(input_dim + self.c_size, num_hidden)

    def forward(self, context_x, target_x):
        encoder_input = self.input_projection(context_x)  # (bs,nc,s+x)--> (bs,nc,num_hidden)
        query = self.target_projection(target_x)  # (bs,nt,s+x)--> (bs,nt,num_hidden)
        keys = self.context_projection(context_x)  # (bs,nc,s+x)--> (bs,nc,num_hidden)

        for attention in self.cross_attentions:  # cross attention layer
            query, _ = attention(keys, encoder_input, query)  # (bs,nt,hidden_number)

        return query


class Decoder(nn.Module):
    """
    Dencoder
    """

    def __init__(self, x_size, y_size, c_size, num_hidden):
        super(Decoder, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.c_size = c_size
        self.num_hidden = num_hidden
        
        self.attribute = Linear(self.x_size, int(self.num_hidden / 4))
        self.location = Linear(self.c_size, int(self.num_hidden / 4))
        self.decoder1 = nn.Sequential(
            nn.Linear(2 * self.num_hidden + int(self.num_hidden / 2), num_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_hidden, 1),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.num_hidden + int(self.num_hidden / 2), num_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_hidden, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, target_x, r, w, v):
        """ context_x : (batch_size, n_context, x_size)
            context_y : (batch_size, n_context, y_size)
            target_x : (batch_size, n_target, x_size)
        """

        bs, nt, x_size = target_x.shape  # (bs,nt, x_size)
        t_x = self.attribute(target_x[..., self.c_size:])
        s_x = self.location(target_x[..., 0:self.c_size])
        z_tx = torch.cat([t_x, s_x], dim=-1)

        z1_tx = torch.cat([torch.cat([w, z_tx], dim=-1), r], dim=-1)  # cat(x*,s*,r,w)
        z1_tx = z1_tx.view((bs * nt, 2 * self.num_hidden + int(self.num_hidden / 2)))
        decoder1 = self.decoder1(z1_tx)
        decoder1 = decoder1.view((bs, nt, 1))  # (bs, nt, 1)
        mu = decoder1[:, :, 0]

        z2_tx = torch.cat([z_tx, v], dim=-1)  # cat(x*,s*,v)
        z2_tx = z2_tx.view((bs * nt, self.num_hidden + int(self.num_hidden / 2)))
        decoder2 = self.decoder2(z2_tx)
        decoder2 = decoder2.view((bs, nt, 1))  # (bs, nt,1)
        log_sigma = decoder2[:, :, 0]


        sigma = 0.1 + 0.9 * self.softplus(log_sigma)  # variance  sigma=0.1+0.9*log(1+exp(log_sigma))
        return mu, sigma


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

        self.x_size = len(self.dataset_info["op_dic"]) + len(self.dataset_info["non_eu_col"])
        self.y_size = 1
        self.c_size = len(list(self.dataset_info["eu_col"].keys()))
        self.num_hidden = settings["num_hidden"]

        self.determine = DeterministicEncoder(self.x_size, self.y_size, self.c_size, self.num_hidden)
        self.location = LocationEncoder(self.x_size, self.y_size, self.c_size, self.num_hidden)
        self.variance = VarianceEncoder(self.x_size, self.y_size, self.c_size, self.num_hidden)
        self.decoder = Decoder(self.x_size, self.y_size, self.c_size, self.num_hidden)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=settings['nn_lr'])
        self.scheduler = None

    def forward(self, batch):
        inputs, coords, targets, input_lenths = batch

        # print(inputs.size(), coords.size(), targets.size())

        context_x = torch.cat([coords[:, 1:, :], inputs[:, 1:, :]], dim=-1)
        context_y = targets[:, 1:, :]
        target_x = torch.cat([coords[:, :1, :], inputs[:, :1, :]], dim=-1)
        target_y = targets[:, :1, :]

        # print(context_x.size(), context_y.size(), target_x.size(), target_y.size())

        r = self.determine(context_x, context_y, target_x)  # mean-attribute encoder

        # print('self.determine through!')
        
        w = self.location(context_x, context_y,target_x)  # mean-location encoder

        # print('self.location through!')
        
        v = self.variance(context_x, target_x)   # variance encoder

        # print('self.variance through!')
        
        mu, sigma = self.decoder(target_x, r, w, v)  # decoder

        # print('self.decoder through!')

        # print(mu.size(), sigma.size())

        return (mu, sigma), target_y

    def loss_func(self, model_output, target):
        mu, sigma = model_output

        bs = mu.shape[0]
        nt = mu.shape[1]

        for i in range(bs):
            dist1 = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu[i], covariance_matrix=torch.diag(sigma[i]))

            log_prob = dist1.log_prob(target[i])
            loss = -log_prob/nt  # torch.mean(log_prob)

        loss = loss / bs
        return loss, mu, target
