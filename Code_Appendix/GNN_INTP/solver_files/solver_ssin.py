import time
import random
import json
import os
import math
import numbers
import copy

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
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
    "model": "SSIN",
    'masked_lm_prob': 0.2,
    'd_model': 16,
    'd_inner': 256,
    'n_head': 2,
    'n_layers': 3,
    'dropout': 0.1,

    # Training settings
    'seed': 1,
    'full_batch': 64,
    'real_batch': 32,
    'epoch': 100,
    'trans_lr': 1e-3,
    'nn_lr': 1e-3,
    'es_mindelta': 0.0,
    'es_endure': 3,
}


class ScheduledOptim(_LRScheduler):
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps=4000):
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        super(ScheduledOptim, self).__init__(optimizer)

    def step(self):
        self.n_steps += 1

        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        lr_scale = (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        
        lr = self.lr_mul * lr_scale
        self._last_lr = lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        
def gelu(x):
   return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TwoLayerFCN(nn.Module):
    def __init__(self, feat_dim, n_hidden1, n_hidden2):
        super().__init__()
        self.feat_dim = feat_dim
        self.linear_1 = nn.Linear(feat_dim, n_hidden1)
        self.linear_2 = nn.Linear(n_hidden1, n_hidden2)

    def forward(self, in_vec, non_linear=False):
        """pos_vec: absolute position vector, n * feat_dim"""
        assert in_vec.shape[-1] == self.feat_dim, f"in_vec.shape: {in_vec.shape}, feat_dim:{self.feat_dim}"

        if non_linear:
            mid_emb = F.relu(self.linear_1(in_vec))
        else:
            mid_emb = self.linear_1(in_vec)

        out_emb = self.linear_2(mid_emb)
        return out_emb


class NewRelativeEncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, d_pos, dropout=0.1, temperature=None):
        super(NewRelativeEncoderLayer, self).__init__()
        if temperature is None:
            temperature = d_k ** 0.5

        self.slf_attn = NewRelativeMultiHeadAttention(n_head, d_model, d_k, d_v, d_pos, temperature, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, pos_mat, attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, pos_mat, mask=attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class RelativePosition(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, pos_mat):
        """pos_mat: relative position matrix, b * n * n * pos_dim"""
        assert pos_mat.shape[1] == pos_mat.shape[2]
        # all seq share one relative positional matrix
        n_element = pos_mat.shape[1]
        pos_dim = pos_mat.shape[-1]
        positions = pos_mat.view(-1, pos_dim)
        pos_embeddings = self.linear_2(self.linear_1(positions))

        # [sz_b x len_q x len_q x d_v/d_k]
        return pos_embeddings.view(-1, n_element, n_element, self.out_dim)  # added: batch_size dim


# ----------------------- New versions of relative position -----------------------
class NewRelativeMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, pos_dim, temperature, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.relative_position = RelativePosition(pos_dim, d_k, d_k)
        self.attention = RelativeScaledDotProductAttention(temperature=temperature)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, pos_mat, mask=None):
        d_k, d_v, d_model, n_head = self.d_k, self.d_v, self.d_model, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # generate the spatial relative position embeddings (SRPEs)
        a_k = self.relative_position(pos_mat)

        if mask is not None:  # used to achieve Shielded Attention
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, a_k, d_k, d_v, n_head, mask=mask)

        # Transpose to move the head dimension back: sz_b x len_q x n_head x dv
        # Combine the last two dimensions to concatenate all the heads together: sz_b x len_q x (n_head*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class RelativeScaledDotProductAttention(nn.Module):
    ''' attn: sum over element-wise product of three vectors'''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, a_k, d_k, d_v, n_head, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Transpose for attention dot product: sz_b x n_head x len_q x dv
        # Separate different heads: sz_b x len_q x n_head x dv
        r_q1, r_k1, r_v1 = q.view(sz_b, len_q, n_head, d_k).permute(0, 2, 1, 3), \
                           k.view(sz_b, len_q, n_head, d_k).permute(0, 2, 1, 3), \
                           v.view(sz_b, len_v, n_head, d_v).permute(0, 2, 1, 3)

        # r_q1: [sz_b, n_head, len_q, 1, d_k], r_k1: [sz_b, n_head, 1, len_q, d_k]
        attn1 = torch.mul(r_q1.unsqueeze(2), r_k1.unsqueeze(3))
        # attn1: [sz_b, n_head, len_q, len_q, d_k], a: [sz_b, len_q, len_q, d_k]
        attn = torch.sum(torch.mul(attn1, a_k.unsqueeze(1)), -1)
        attn = attn / self.temperature  # [sz_b x n_head x len_q x len_k]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, r_v1)

        return output, attn


class INTP_Model(nn.Module):
    def __init__(self, settings, device):
        super(INTP_Model, self).__init__()
        
        self.settings = settings
        self.device = device

        origin_path = f"./Datasets/{settings['dataset']}/"
        with open(origin_path + f'meta_data.json', 'r') as f:
            self.dataset_info = json.load(f)

        self.d_model = self.settings['d_model']
        self.d_pos = 2

        if len(self.dataset_info["eu_col"]) == 2:
            self.f_dim = len(self.dataset_info["op_dic"]) + 1
        elif len(self.dataset_info["eu_col"]) == 3:
            self.f_dim = len(self.dataset_info["op_dic"]) + 2
        
        self.feature_enc = TwoLayerFCN(self.f_dim, self.d_model, self.d_model).to(self.device)
        self.dropout = nn.Dropout(p=self.settings['dropout']).to(self.device)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6).to(self.device)

        self.layer_stack = nn.ModuleList(
            [NewRelativeEncoderLayer(self.d_model, self.settings['d_inner'], self.settings['n_head'], self.d_model, self.d_model, self.d_pos, dropout=self.settings['dropout'], temperature=None) for _ in range(self.settings['n_layers'])]
        ).to(self.device)

        self.linear = nn.Linear(self.d_model, self.d_model).to(self.device)
        self.activ2 = gelu
        self.decoder = TwoLayerFCN(self.d_model, self.d_model, 1).to(self.device)
        
        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = ScheduledOptim(self.optimizer, 0.05, self.settings["d_model"], 1200)

    # @autocast()
    def forward(self, batch):
        masked_seqs, masked_idxs, masked_lbls, masked_lws, attn_masks, mean_values, std_values, dams = batch

        feat_seq = masked_seqs    # shape: [b, max_seq_len, f_dim]
        r_pos_mat = dams     # shape: [b, max_seq_len, max_seq_len, 2]
        masked_pos = masked_idxs     # shape: [b, max_pred_per_seq]
        attn_mask = attn_masks

        enc_output = self.feature_enc(feat_seq)
        enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)    # shape: [b, max_seq_len, d_model]
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, r_pos_mat, attn_mask=attn_mask)    # shape: [b, max_seq_len, d_model]

        h_masked_1 = enc_output[torch.arange(enc_output.size(0)).unsqueeze(1), masked_pos]
        h_masked_2 = self.layer_norm(self.activ2(self.linear(h_masked_1)))
        dec_output = self.decoder(h_masked_2)  # [batch_size, max_pred, n_vocab]
        
        return (dec_output, masked_lws, mean_values, std_values), masked_lbls

    
    def loss_func(self, model_output, masked_lbls):
        dec_output, masked_label_weights, mean_values, std_values = model_output

        targets = masked_lbls.unsqueeze(-1)
        per_example_loss = torch.nn.MSELoss(reduction="none")(dec_output, targets)
        
        numerator = torch.sum(per_example_loss.squeeze() * masked_label_weights)
        denominator = torch.sum(masked_label_weights) + 1e-10
        loss = numerator / denominator

        # print('*', dec_output.size(), targets.size())
        
        valid_output = dec_output * (std_values[:, 0:1].unsqueeze(1) + 1e-12) + mean_values[:, 0:1].unsqueeze(1)
        valid_labels = targets * (std_values[:, 0:1].unsqueeze(1) + 1e-12) + mean_values[:, 0:1].unsqueeze(1)

        # print('**', valid_output.size(), valid_labels.size())
        
        valid_output = valid_output[masked_label_weights == 1]
        valid_labels = valid_labels[masked_label_weights == 1]

        # print('***', valid_output.size(), valid_labels.size())

        return loss, valid_output, valid_labels
