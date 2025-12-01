# Transformer implementation from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# GELU implementation from https://github.com/karpathy/minGPT

import time
import random
import json
import os
import math
import numbers
import copy

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange, repeat
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from Datasets import aio_dataloader as dl
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
    "model": "Transformer_Stage",
    'embedding_dim': 256,
    'num_head': 2,
    'num_layers_a': 2,
    'num_layers_k': 2,
    'output_hidden_layers': 0,
    'dropout': 0.0,

    # Training settings
    'seed': 1,
    'full_batch': 64,
    'real_batch': 64,
    'epoch': 100,
    'trans_lr': 1e-3,
    'nn_lr': 1e-3,
    'es_mindelta': 0.0,
    'es_endure': 3,
}


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, d_model)
    
    def forward(self, x):
        x = self.dropout(nn.LeakyReLU(0.1)(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.eps = eps
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask, dropout):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    # print(scores.size())
    # print(mask.size())
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, k_mode):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model
        self.h = nhead
        self.k_mode = k_mode
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model + k_mode, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(nhead * d_model, d_model)
    
    def forward(self, q, k, v, mask):
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).unsqueeze(-2).repeat(1, 1, self.h, 1)
        q = self.q_linear(q).unsqueeze(-2).repeat(1, 1, self.h, 1)
        v = self.v_linear(v).unsqueeze(-2).repeat(1, 1, self.h, 1)
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.h * self.d_model)
        output = self.out(concat)
    
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, k_mode):
        super().__init__()
        self.k_mode = k_mode
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(d_model, nhead, dropout, k_mode)

        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        
    def forward(self, x, env, mask):
        x2 = self.norm_1(x)
        # x2 = x
        
        # if env is not None:
        #     print(f"x2: {x2.size()}")
        #     print(f"env: {env.size()}")
        #     print(f"cb: {torch.concat([x2, env.unsqueeze(1).repeat(1, x2.size(1), 1)], dim=2).size()}")
        
        if self.k_mode == 0:
            x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        else:
            x = x + self.dropout_1(self.attn(x2, torch.concat([x2, env.unsqueeze(1).repeat(1, x2.size(1), 1)], dim=2), x2, mask))
        
        x2 = self.norm_2(x)
        # x2 = x
        x = x + self.dropout_2(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, k_mode):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.layers = get_clones(EncoderLayer(d_model, nhead, dim_feedforward, dropout, k_mode), num_encoder_layers)
        self.norm = Norm(d_model)
        
    def forward(self, src, env, mask):
        for i in range(self.num_encoder_layers):
            src = self.layers[i](src, env, mask)
        return self.norm(src)
        # return src


def length_to_mask(lengths, total_len, device):
    max_len = total_len
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1)
    return mask.unsqueeze(-2)


# Returns the closest number that is a power of 2 to the given real number x
def closest_power_of_2(x):
    return 2 ** round(math.log2(x))


# Returns a list of n numbers that are evenly spaced between a and b.
def evenly_spaced_numbers(a, b, n):
    if n == 1:
        return [(a+b)/2]
    step = (b-a)/(n-1)
    return [a + i*step for i in range(n)]
    
    
# generate a V-shape MLP as torch.nn.Sequential given input_size, output_size, and layer_count(only linear layer counted)
def generate_sequential(a, b, n, p):
    layer_sizes = evenly_spaced_numbers(a, b, n)
    layer_sizes = [int(layer_sizes[0])] + [int(closest_power_of_2(x)) for x in layer_sizes[1:-1]] + [int(layer_sizes[-1])]
    
    layers = []
    for i in range(n-1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i == 0:
            layers.append(nn.LeakyReLU(0.1))
        elif 0 < i < n-2:
            layers.append(nn.LeakyReLU(0.1))
            # layers.append(torch.nn.Dropout(p))
    
    model = torch.nn.Sequential(*layers)
    return model


def lr_schedule(epoch):
    warmup_steps = 1000
    if epoch < warmup_steps:
        return float(epoch) / float(warmup_steps)
    else:
        return 1.0


class INTP_Model(nn.Module):
    def __init__(self, settings, device):
        super(INTP_Model, self).__init__()
        
        self.settings = settings
        self.device = device

        origin_path = f"./Datasets/{settings['dataset']}/"
        with open(origin_path + f'meta_data.json', 'r') as f:
            self.dataset_info = json.load(f)

        settings['q_dim'] = len(self.dataset_info["eu_col"]) + len(self.dataset_info["non_eu_col"])
        settings['k_dim'] = settings['q_dim'] + 1 + len(self.dataset_info["op_dic"])
        settings['a_dim'] = settings['k_dim']
        settings['feedforward_dim'] = settings['embedding_dim'] * 2
        settings['q_embedding_dim'] = settings['embedding_dim'] // 4

        self.a_embedding_layer = torch.nn.Linear(settings['a_dim'], settings['embedding_dim']).to(self.device)

        self.k_transformer = Encoder(
            d_model=settings['embedding_dim'], nhead=settings['num_head'], num_encoder_layers=settings['num_layers_k'], 
            dim_feedforward=settings['feedforward_dim'], dropout=settings['dropout'], k_mode=0
        ).to(self.device)

        self.output = generate_sequential(settings['embedding_dim'], 1, settings['output_hidden_layers']+3, settings['dropout']).to(self.device) 
        
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        self.optimizer = optim.AdamW([
                {'params': self.a_embedding_layer.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 1e-4},
            
                {'params': self.k_transformer.parameters(), 'lr': settings['trans_lr'], 'weight_decay': 1e-4},
            
                {'params': self.output.parameters(), 'lr': settings['nn_lr'], 'weight_decay': 1e-4},
            ], 
            betas=(0.9, 0.95), eps=1e-8
        )
        self.scheduler = None

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def log_prob(self, mu, std, value):
        # compute the variance
        var = (std ** 2)
        log_scale = math.log(std) if isinstance(std, numbers.Real) else std.log()
        return -((value - mu) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def kl_divergence(self, z, mu, std, tgt, c, encode):
        if encode:
            mu_full = c.repeat(1, int(mu.size(1)) // 2)
            std_full = torch.ones_like(std)
        else:
            mu_full = torch.zeros_like(mu)
            std_full = torch.ones_like(std)

        log_qzx = self.log_prob(mu, std, z)
        log_pz = self.log_prob(mu_full, std_full, z)
        kl = (log_qzx - log_pz)
        kl = kl.mean(-1)
        return kl.unsqueeze(1)

    # @autocast()
    def forward(self, batch):
        q_tokens, known_lenths, auxil_lenths, input_series, answers = batch

        in_series = torch.concat([q_tokens.unsqueeze(1), input_series], dim=1)

        a_tokens_emb = self.a_embedding_layer(in_series)
        k_attention_mask = length_to_mask(known_lenths+auxil_lenths+1, a_tokens_emb.size(1), self.device)
        token_k = self.k_transformer(a_tokens_emb, None, k_attention_mask)[:, 0, :]
        
        output = self.output(token_k)

        target = answers.unsqueeze(1).to(self.device)

        model_output = output
        
        return model_output, target

    
    def loss_func(self, model_output, target):
        output = model_output
        
        # reconstruction loss (batch average with mask)
        recon_loss = torch.nn.L1Loss(reduction='mean')(output, target)

        if self.training:
            elbo = recon_loss
        else:
            elbo = recon_loss
        
        return elbo, output, target
