import json
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, knn_graph

import dgl
import dgl.function as fn
from scipy import sparse as sp
from scipy.sparse.linalg import norm 

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

    # # Model settings
    # "model": "LSPE",
    # 'k': 5,
    # "pos_enc_dim": 32,
    # "pe_init": 'rand_walk',
    # 'conv_dim': 256,
    # 'L': 2,
    # 'alpha_loss': 1,
    # 'lambda_loss': 0.1,

    # 'in_feat_dropout': 0.0,
    # 'dropout': 0.0,

    # Model settings
    "model": "LSPE2",
    "postfix": "32_5_sep",
    'k': 5,
    "pos_enc_dim": 32,
    "pe_init": 'rand_walk',
    'conv_dim': 59,
    'L': 16,
    'alpha_loss': 1e-4,
    'lambda_loss': 1,
    'in_feat_dropout': 0.0,
    'dropout': 0.0,

    # Training settings
    'seed': 1,
    'full_batch': 64,
    'real_batch': 16,
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


class GatedGCNLSPELayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, use_lapeig_loss=False, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_lapeig_loss = use_lapeig_loss
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A1 = nn.Linear(input_dim*2, output_dim, bias=True)
        self.A2 = nn.Linear(input_dim*2, output_dim, bias=True)
        self.B1 = nn.Linear(input_dim, output_dim, bias=True)
        self.B2 = nn.Linear(input_dim, output_dim, bias=True)
        self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.C1 = nn.Linear(input_dim, output_dim, bias=True)
        self.C2 = nn.Linear(input_dim, output_dim, bias=True)
        
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        # self.bn_node_p = nn.BatchNorm1d(output_dim)

    def message_func_for_vij(self, edges):
        hj = edges.src['h'] # h_j
        pj = edges.src['p'] # p_j
        vij = self.A2(torch.cat((hj, pj), -1))
        return {'v_ij': vij} 
    
    def message_func_for_pj(self, edges):
        pj = edges.src['p'] # p_j
        return {'C2_pj': self.C2(pj)}
       
    def compute_normalized_eta(self, edges):
        return {'eta_ij': edges.data['sigma_hat_eta'] / (edges.dst['sum_sigma_hat_eta'] + 1e-6)} # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
      
    def forward(self, g, h, p, e, snorm_n):   

        with g.local_scope():
        
            # for residual connection
            h_in = h 
            p_in = p 
            e_in = e 

            # For the h's
            g.ndata['h']  = h 
            g.ndata['A1_h'] = self.A1(torch.cat((h, p), -1)) 
            # self.A2 being used in message_func_for_vij() function
            g.ndata['B1_h'] = self.B1(h)
            g.ndata['B2_h'] = self.B2(h) 

            # For the p's
            g.ndata['p'] = p
            g.ndata['C1_p'] = self.C1(p)
            # self.C2 being used in message_func_for_pj() function

            # For the e's
            g.edata['e']  = e 
            g.edata['B3_e'] = self.B3(e) 

            #--------------------------------------------------------------------------------------#
            # Calculation of h
            g.apply_edges(fn.u_add_v('B1_h', 'B2_h', 'B1_B2_h'))
            g.edata['hat_eta'] = g.edata['B1_B2_h'] + g.edata['B3_e']
            g.edata['sigma_hat_eta'] = torch.sigmoid(g.edata['hat_eta'])
            g.update_all(fn.copy_e('sigma_hat_eta', 'm'), fn.sum('m', 'sum_sigma_hat_eta')) # sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.compute_normalized_eta) # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.message_func_for_vij) # v_ij
            g.edata['eta_mul_v'] = g.edata['eta_ij'] * g.edata['v_ij'] # eta_ij * v_ij
            g.update_all(fn.copy_e('eta_mul_v', 'm'), fn.sum('m', 'sum_eta_v')) # sum_j eta_ij * v_ij
            g.ndata['h'] = g.ndata['A1_h'] + g.ndata['sum_eta_v']

            # Calculation of p
            g.apply_edges(self.message_func_for_pj) # p_j
            g.edata['eta_mul_p'] = g.edata['eta_ij'] * g.edata['C2_pj'] # eta_ij * C2_pj
            g.update_all(fn.copy_e('eta_mul_p', 'm'), fn.sum('m', 'sum_eta_p')) # sum_j eta_ij * C2_pj
            g.ndata['p'] = g.ndata['C1_p'] + g.ndata['sum_eta_p']

            #--------------------------------------------------------------------------------------#

            # passing towards output
            h = g.ndata['h'] 
            p = g.ndata['p']
            e = g.edata['hat_eta'] 

            # GN from benchmarking-gnns-v1
            h = h * snorm_n
            
            # batch normalization  
            if self.batch_norm:
                h = self.bn_node_h(h)
                e = self.bn_node_e(e)
                # No BN for p

            # non-linear activation
            h = F.relu(h) 
            e = F.relu(e) 
            p = torch.tanh(p)

            # residual connection
            if self.residual:
                h = h_in + h 
                p = p_in + p
                e = e_in + e 

            # dropout
            h = F.dropout(h, self.dropout, training=self.training)
            p = F.dropout(p, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

            return h, p, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


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

        self.embedding_p = nn.Linear(settings["pos_enc_dim"], settings["conv_dim"]).to(self.device)
        self.embedding_h = nn.Linear(settings["num_features_in"], settings["conv_dim"]).to(self.device)
        self.embedding_e = nn.Linear(1, settings["conv_dim"]).to(self.device)
        self.in_feat_dropout = nn.Dropout(settings["in_feat_dropout"]).to(self.device)

        self.layers = nn.ModuleList([
            GatedGCNLSPELayer(settings["conv_dim"], settings["conv_dim"], settings["dropout"], True, residual=True).to(self.device) for _ in range(settings["L"]-1) 
        ]) 
        self.layers.append(GatedGCNLSPELayer(self.settings["conv_dim"], settings["conv_dim"], settings["dropout"], True, residual=True).to(self.device))

        self.MLP_layer = MLPReadout(settings["conv_dim"], settings['num_features_out']).to(self.device)
        self.p_out = nn.Linear(settings["conv_dim"], settings["pos_enc_dim"]).to(self.device)
        self.Whp = nn.Linear(settings["conv_dim"] + settings["pos_enc_dim"], settings["conv_dim"]).to(self.device)

        self.g = None
        
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        # self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=5e-4, lr=settings['nn_lr'])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=settings['nn_lr'])
        self.scheduler = None

    def forward(self, batch):
        g, targets, snorm_n = batch
        g, targets, snorm_n = g.to(self.device), targets.to(self.device), snorm_n.to(self.device)

        h = g.ndata['feat'].to(self.device)
        e = g.edata['feat'].view(-1, 1).to(self.device)
        p = g.ndata['pos_enc'].to(self.device)

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        p = self.embedding_p(p)
        e = self.embedding_e(e)

        for conv in self.layers:
            h, p, e = conv(g, h, p, e, snorm_n)

        g.ndata['h'] = h

        p = self.p_out(p)
        g.ndata['p'] = p
        means = dgl.mean_nodes(g, 'p')
        batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
        p = p - batch_wise_p_means

        g.ndata['p'] = p
        g.ndata['p2'] = g.ndata['p']**2
        norms = dgl.sum_nodes(g, 'p2')
        norms = torch.sqrt(norms)            
        batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
        p = p / batch_wise_p_l2_norms
        g.ndata['p'] = p
    
        hp = self.Whp(torch.cat((g.ndata['h'],g.ndata['p']),dim=-1))
        g.ndata['h'] = hp

        # readout
        # print(f"g.ndata['h'] size: {g.ndata['h'].size()}")

        num_nodes_per_graph = g.batch_num_nodes()
        first_node_indices = [0, ]
        for i in range(1, len(num_nodes_per_graph)):
            first_node_indices.append(sum(num_nodes_per_graph[:i]).item())

        # print(first_node_indices)
        # print(g.ndata['h'].size())
        
        hg = g.ndata['h'][first_node_indices]

        # first_node_ids = []
        # for g_item in g:
        #     first_node_ids.append(g_item.nodes()[0].item())
        # print(first_node_ids)
        # first_node_hs = dgl.batched_select(batch_graph, first_node_ids, 'n')['h']
        
        # print(f"hg size: {hg.size()}")

        return (self.MLP_layer(hg), g), targets

    def loss_func(self, model_output, target):
        scores, g = model_output

        # print(f"scores size: {scores.size()}")
        # print(f"target size: {target.size()}")

        loss_a = nn.L1Loss()(scores, target)

        # n = g.number_of_nodes()
        # # Laplacian 
        # A = g.adj_external(scipy_fmt="csr")
        # N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        # L = sp.eye(n) - N * A * N
        # p = g.ndata['p']
        # pT = torch.transpose(p, 1, 0)
        # loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(self.device)), p))
        # # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
        # bg = dgl.unbatch(g)
        # batch_size = len(bg)
        # P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
        # PTP_In = P.T * P - sp.eye(P.shape[1])
        # loss_b_2 = torch.tensor(norm(PTP_In, 'fro')**2).float().to(self.device)
        # loss_b = ( loss_b_1 + self.settings["lambda_loss"] * loss_b_2 ) / ( self.settings["pos_enc_dim"] * batch_size * n) 

        # del bg, P, PTP_In, loss_b_1, loss_b_2

        # loss = loss_a + self.settings["alpha_loss"] * loss_b

        loss = loss_a

        return loss, scores, target
