import json
import pickle
import concurrent.futures
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import knn_graph
import dgl
from scipy import sparse as sp

import support_functions


def encode_and_bind(original_dataframe, feature_to_encode, possible_values):
    enc_df = original_dataframe.copy()
    for value in possible_values:
        enc_df.loc[:, feature_to_encode + '_' + str(value)] = (enc_df[feature_to_encode] == value).astype(int)
    res = enc_df.drop([feature_to_encode], axis=1)
    return res


def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """

    # Laplacian
    A = g.adj_external(scipy_fmt="csr")
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['eigvec'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))

    return g


def init_positional_encoding(g, pos_enc_dim, type_init):
    """
        Initializing positional encoding with RWPE
    """
    
    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adj_external(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pos_enc'] = PE  
    
    return g


class IntpDataset(Dataset):
    def __init__(self, settings, mask_distance, call_name):
        # save init parameters
        self.settings = settings
        self.mask_distance = mask_distance
        self.call_name = call_name

        # load dataset info
        self.origin_path = f"./Datasets/{settings['dataset']}/"
        with open(self.origin_path + f'meta_data.json', 'r') as f:
            self.dataset_info = json.load(f)
        with open(self.origin_path + f"Folds_Info/norm_{settings['fold']}_{settings['holdout']}.json", 'r') as f:
            self.dic_op_minmax = json.load(f)
        with open(self.origin_path + f"Folds_Info/divide_set_{settings['fold']}_{settings['holdout']}.info", 'rb') as f:
            divide_set = pickle.load(f)

        # load file list
        if call_name == 'train':
            call_scene_list = divide_set[0]
        elif call_name == 'test':
            call_scene_list = divide_set[1]
        elif call_name == 'eval':
            call_scene_list = divide_set[2]
        if settings['debug'] and len(call_scene_list) > 1000:
            call_scene_list = call_scene_list[:1000]

        # do op filtering and normalization in the parallel fashion
        self.total_df_dict = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
            for future in concurrent.futures.as_completed(futures):
                file_name, file_content = future.result()
                self.total_df_dict[file_name] = file_content

        self.call_list = []
        target_op_range = self.dataset_info["holdouts"][str(self.settings['holdout'])][self.call_name]
        for scene in call_scene_list:
            df = self.total_df_dict[scene]
            target_row_index_list = list(df[df['op'].isin(target_op_range)].index)
            target_row_index_list.sort()
            # if call_name == 'test' and len(target_row_index_list) >= 2:
            #     target_row_index_list = target_row_index_list[:2]
            for index in target_row_index_list:
                self.call_list.append([scene, index])
        tail_index = (len(self.call_list) // 256) * 256
        self.call_list = self.call_list[:tail_index]

        print(f"Length of df dict: {len(list(self.total_df_dict.keys()))}")
        print(f"Length of call list: {len(self.call_list)}")

    
    def __len__(self):
        return len(self.call_list)


    def norm(self, d):
        d_list = []
        for op in d['op'].unique():
            d_op = d[d['op']==op].copy()
            if op in self.dataset_info["tgt_logical"]:
                op_norm = self.dataset_info["tgt_op"]
            else:
                op_norm = op
            d_op['Result_norm'] = (d_op['Result'] - self.dic_op_minmax[op_norm][0]) / (self.dic_op_minmax[op_norm][1] - self.dic_op_minmax[op_norm][0])
            d_list.append(d_op)
        return pd.concat(d_list, axis=0, ignore_index=False).drop(columns=['Result'])


    def process_child(self, filename):
        df = pd.read_csv(self.origin_path + 'Dataset_Separation/' + filename, sep=';')
        # drop everything in bad quality
        df = df[df['Thing']>=self.dataset_info['lowest_rank']]
        # drop everything with non-whitelisted op
        op_whitelist = list(self.dataset_info["op_dic"].keys())
        for holdout in self.dataset_info["holdouts"].keys():
            op_whitelist = op_whitelist + self.dataset_info["holdouts"][holdout]["train"] + self.dataset_info["holdouts"][holdout]["test"] + self.dataset_info["holdouts"][holdout]["eval"]
        op_whitelist = list(set(op_whitelist))
        df = df[df['op'].isin(op_whitelist)]
        # normalize all values (coordinates will be normalized later)
        df = self.norm(d=df)
        return filename, df

    def distance_matrix(self, obs, interp):
        n_obs, n_dim_obs = obs.shape
        n_interp, n_dim_interp = interp.shape
        # Expand dimensions to broadcast
        obs = obs[:, np.newaxis, :]
        interp = interp[np.newaxis, :, :]
        # Calculate differences along each dimension
        diffs = obs - interp
        # Calculate squared distances along each dimension
        squared_diffs = diffs ** 2
        # Sum along dimensions to get squared distance
        squared_distances = np.sum(squared_diffs, axis=2)
        # Take square root to get actual distance
        distances = np.sqrt(squared_distances)
        return distances
    
    
    def idw_interpolation(self, obs_coords, values, interp_coords, p=2):
        dist = self.distance_matrix(obs_coords, interp_coords)
        # In IDW, weights are 1 / distance
        weights = 1.0 / (dist + 1e-12) ** p
        # Make weights sum to one
        weights /= np.sum(weights, axis=0, keepdims=True)
        # Multiply the weights for each interpolated point by all observed Z-values
        interpolated_values = np.dot(values, weights)
        return interpolated_values


    def norm_fcol(self, df):
        rtn = df.copy()
        # Norm other columns
        for col, (min_val, max_val) in self.dataset_info["eu_col"].items():
            rtn[col] = (rtn[col] - min_val) / (max_val - min_val)
        for col, (min_val, max_val) in self.dataset_info["non_eu_col"].items():
            rtn[col] = (rtn[col] - min_val) / (max_val - min_val)
        return rtn


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data item
        df = self.total_df_dict[self.call_list[idx][0]]
        target_row_index = self.call_list[idx][1]
        
        # keep target row
        target_row = pd.DataFrame(df.loc[target_row_index]).transpose()
        target_row = target_row.apply(pd.to_numeric, errors='ignore')
        df = df.drop(target_row_index)
        # clean unrelated labels
        keep_op_range = list(set(
            list(self.dataset_info["op_dic"].keys()) + 
            self.dataset_info["holdouts"][str(self.settings['holdout'])]["train"]
        ))
        df = df[df['op'].isin(keep_op_range)]
        
        while True:
            if self.call_name == 'eval':
                data_augmentation_option = 'dst'
                this_mask = self.mask_distance
                df_filtered = df.loc[(abs(df['Longitude'] - target_row['Longitude'].values[0]) + abs(df['Latitude'] - target_row['Latitude'].values[0])) >= this_mask, :].copy()
            else:
                df_filtered = df

            df_filtered.loc[df_filtered['op'].isin(self.dataset_info["tgt_logical"]), 'op'] = self.dataset_info["tgt_op"]
            target_row.loc[target_row['op'].isin(self.dataset_info["tgt_logical"]), 'op'] = self.dataset_info["tgt_op"]

            df_known = df_filtered[df_filtered['op']==self.dataset_info["tgt_op"]].copy()
            df_auxil = df_filtered[df_filtered['op']!=self.dataset_info["tgt_op"]].copy()
            # check and quit loop
            if len(df_known) >= 1 and len(df_auxil) >= 1:
                break

        # turn dataframe to torch tensor according to models
        rtn = getattr(self, f"to_torch_{self.settings['model'].lower()}", None)(df_known, df_auxil, target_row)
        return rtn


    def collate_fn(self, examples):
        return getattr(self, f"collate_fn_{self.settings['model'].lower()}", None)(examples)


    def to_torch_gnn(self, df_known, df_auxil, target_row):
        df_known = self.norm_fcol(df_known)
        df_auxil = self.norm_fcol(df_auxil)
        target_row = self.norm_fcol(target_row)

        graph_candidates = pd.concat([target_row, df_known], axis=0, ignore_index=True)
        
        coords = torch.from_numpy(graph_candidates[list(self.dataset_info["eu_col"].keys())].values).float()
        # coords = coords - coords[0, :]
        
        answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()

        full_df = pd.concat([df_known, df_auxil], axis=0, ignore_index=True)
        aggregated_df = full_df.groupby(list(self.dataset_info["eu_col"].keys()) + ['op']).mean().reset_index()
        features = torch.zeros((len(graph_candidates), len(self.dataset_info["op_dic"]) + 1))
        possible_values = list(self.dataset_info["op_dic"].keys())
        possible_values.sort()
        for op in possible_values:
            aggregated_df_op = aggregated_df[aggregated_df['op']==op]
            interpolated_grid = torch.zeros((len(graph_candidates), 1))
            if len(aggregated_df_op) != 0:
                c = aggregated_df_op[list(self.dataset_info["eu_col"].keys())].values
                v = aggregated_df_op['Result_norm'].values
                ci = graph_candidates[list(self.dataset_info["eu_col"].keys())].values
                interpolated_values = self.idw_interpolation(c, v, ci)
                interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(graph_candidates), 1))
            features[:, self.dataset_info["op_dic"][op]:self.dataset_info["op_dic"][op]+1] = interpolated_grid
        
        conditions = graph_candidates['op'] == self.dataset_info["tgt_op"]
        features[:, -1] = torch.from_numpy(np.where(conditions, graph_candidates['Thing'], (self.dataset_info['lowest_rank']-1)/self.dataset_info["non_eu_col"]["Thing"][1]))
        features = features.float()

        features[0, :] = 0

        return features, coords, answers


    def collate_fn_gnn(self, examples):
        input_lenths = torch.tensor([len(ex[0]) for ex in examples if len(ex[0]) > 2])
        x_b = pad_sequence([ex[0] for ex in examples if len(ex[0]) > 2], batch_first=True, padding_value=0.0)
        c_b = pad_sequence([ex[1] for ex in examples if len(ex[1]) > 2], batch_first=True, padding_value=0.0)
        y_b = pad_sequence([ex[2] for ex in examples if len(ex[2]) > 2], batch_first=True, padding_value=0.0)
        
        return x_b, c_b, y_b, input_lenths


    to_torch_gcn = to_torch_gnn
    collate_fn_gcn = collate_fn_gnn
    to_torch_gat = to_torch_gnn
    collate_fn_gat = collate_fn_gnn
    to_torch_gsage = to_torch_gnn
    collate_fn_gsage = collate_fn_gnn

    to_torch_gsage_c = to_torch_gnn
    collate_fn_gsage_c = collate_fn_gnn
    
    to_torch_ksage = to_torch_gnn
    collate_fn_ksage = collate_fn_gnn
    to_torch_pegsage = to_torch_gnn
    collate_fn_pegsage = collate_fn_gnn

    to_torch_fegsage = to_torch_gnn
    collate_fn_fegsage = collate_fn_gnn

    to_torch_fegsage_c = to_torch_gnn
    collate_fn_fegsage_c = collate_fn_gnn
    to_torch_fegsage_ck = to_torch_gnn
    collate_fn_fegsage_ck = collate_fn_gnn

    to_torch_nodeformer = to_torch_gnn
    collate_fn_nodeformer = collate_fn_gnn


    def to_torch_cgnn(self, df_known, df_auxil, target_row):
        df_known = self.norm_fcol(df_known)
        df_auxil = self.norm_fcol(df_auxil)
        target_row = self.norm_fcol(target_row)

        graph_candidates = pd.concat([target_row, df_known], axis=0, ignore_index=True)
        
        coords = torch.from_numpy(graph_candidates[list(self.dataset_info["eu_col"].keys())].values).float()
        c_cen = coords - coords[0, :]
        
        answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()

        full_df = pd.concat([df_known, df_auxil], axis=0, ignore_index=True)
        aggregated_df = full_df.groupby(list(self.dataset_info["eu_col"].keys()) + ['op']).mean().reset_index()
        features = torch.zeros((len(graph_candidates), len(self.dataset_info["op_dic"]) + 1))
        possible_values = list(self.dataset_info["op_dic"].keys())
        possible_values.sort()
        for op in possible_values:
            aggregated_df_op = aggregated_df[aggregated_df['op']==op]
            interpolated_grid = torch.zeros((len(graph_candidates), 1))
            if len(aggregated_df_op) != 0:
                c = aggregated_df_op[list(self.dataset_info["eu_col"].keys())].values
                v = aggregated_df_op['Result_norm'].values
                ci = graph_candidates[list(self.dataset_info["eu_col"].keys())].values
                interpolated_values = self.idw_interpolation(c, v, ci)
                interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(graph_candidates), 1))
            features[:, self.dataset_info["op_dic"][op]:self.dataset_info["op_dic"][op]+1] = interpolated_grid
        
        conditions = graph_candidates['op'] == self.dataset_info["tgt_op"]
        features[:, -1] = torch.from_numpy(np.where(conditions, graph_candidates['Thing'], (self.dataset_info['lowest_rank']-1)/self.dataset_info["non_eu_col"]["Thing"][1]))
        features = features.float()

        features[0, :] = 0

        return features, coords, answers, c_cen


    def collate_fn_cgnn(self, examples):
        input_lenths = torch.tensor([len(ex[0]) for ex in examples if len(ex[0]) > 2])
        x_b = pad_sequence([ex[0] for ex in examples if len(ex[0]) > 2], batch_first=True, padding_value=0.0)
        c_b = pad_sequence([ex[1] for ex in examples if len(ex[1]) > 2], batch_first=True, padding_value=0.0)
        y_b = pad_sequence([ex[2] for ex in examples if len(ex[2]) > 2], batch_first=True, padding_value=0.0)
        cc_b = pad_sequence([ex[3] for ex in examples if len(ex[3]) > 2], batch_first=True, padding_value=0.0)
        
        return x_b, c_b, y_b, cc_b, input_lenths


    to_torch_fegsage_ct = to_torch_cgnn
    collate_fn_fegsage_ct = collate_fn_cgnn
    to_torch_fegsage_cn = to_torch_cgnn
    collate_fn_fegsage_cn = collate_fn_cgnn
    to_torch_fegsage_cna = to_torch_cgnn
    collate_fn_fegsage_cna = collate_fn_cgnn
    to_torch_fegsage_cr = to_torch_cgnn
    collate_fn_fegsage_cr = collate_fn_cgnn
    to_torch_fegsage_cra = to_torch_cgnn
    collate_fn_fegsage_cra = collate_fn_cgnn
    to_torch_fegsage_cra2 = to_torch_cgnn
    collate_fn_fegsage_cra2 = collate_fn_cgnn
    to_torch_fegsage_cc = to_torch_cgnn
    collate_fn_fegsage_cc = collate_fn_cgnn
    to_torch_fegsage_cca = to_torch_cgnn
    collate_fn_fegsage_cca = collate_fn_cgnn
    to_torch_fegsage_ca = to_torch_cgnn
    collate_fn_fegsage_ca = collate_fn_cgnn
    
    to_torch_fegsage_ca_gb = to_torch_cgnn
    collate_fn_fegsage_ca_gb = collate_fn_cgnn
    to_torch_fegsage_ca_gb_na = to_torch_cgnn
    collate_fn_fegsage_ca_gb_na = collate_fn_cgnn
    to_torch_fegsage_ca_gb_ng = to_torch_cgnn
    collate_fn_fegsage_ca_gb_ng = collate_fn_cgnn
    to_torch_fegsage_ca_gb_ns = to_torch_cgnn
    collate_fn_fegsage_ca_gb_ns = collate_fn_cgnn

    to_torch_fegsage_ca_gbd = to_torch_cgnn
    collate_fn_fegsage_ca_gbd = collate_fn_cgnn
    to_torch_fegsage_ca_gbd2 = to_torch_cgnn
    collate_fn_fegsage_ca_gbd2 = collate_fn_cgnn
    to_torch_fegsage_ca_gbd2_wow = to_torch_cgnn
    collate_fn_fegsage_ca_gbd2_wow = collate_fn_cgnn
    to_torch_fegsage_ca_gbd2_wow2 = to_torch_cgnn
    collate_fn_fegsage_ca_gbd2_wow2 = collate_fn_cgnn

    to_torch_fegsage_ca_gbd2_wow2w = to_torch_cgnn
    collate_fn_fegsage_ca_gbd2_wow2w = collate_fn_cgnn
    
    to_torch_fegsage_ca_gbd2_wowc = to_torch_cgnn
    collate_fn_fegsage_ca_gbd2_wowc = collate_fn_cgnn
    to_torch_fegsage_ca_gbd2_wowr = to_torch_cgnn
    collate_fn_fegsage_ca_gbd2_wowr = collate_fn_cgnn
    to_torch_fegsage_ca_gbd2_wowrc = to_torch_cgnn
    collate_fn_fegsage_ca_gbd2_wowrc = collate_fn_cgnn
    to_torch_fegsage_ca_gbd3 = to_torch_cgnn
    collate_fn_fegsage_ca_gbd3 = collate_fn_cgnn

    to_torch_fegsage_ca_gbd_end = to_torch_cgnn
    collate_fn_fegsage_ca_gbd_end = collate_fn_cgnn
    to_torch_fegsage_ca_gbd_end_woc = to_torch_cgnn
    collate_fn_fegsage_ca_gbd_end_woc = collate_fn_cgnn
    to_torch_fegsage_ca_gbd_end_wot = to_torch_cgnn
    collate_fn_fegsage_ca_gbd_end_wot = collate_fn_cgnn
    to_torch_fegsage_ca_gbd_end_na = to_torch_cgnn
    collate_fn_fegsage_ca_gbd_end_na = collate_fn_cgnn

    to_torch_fegsage_ca_gbd_end_refined = to_torch_cgnn
    collate_fn_fegsage_ca_gbd_end_refined = collate_fn_cgnn
    
    to_torch_fegsage_ca2 = to_torch_cgnn
    collate_fn_fegsage_ca2 = collate_fn_cgnn
    to_torch_fegsage_ca_rb = to_torch_cgnn
    collate_fn_fegsage_ca_rb = collate_fn_cgnn
    to_torch_fegsage_ca_rbi = to_torch_cgnn
    collate_fn_fegsage_ca_rbi = collate_fn_cgnn
    to_torch_fegsage_ca_rbit = to_torch_cgnn
    collate_fn_fegsage_ca_rbit = collate_fn_cgnn
    to_torch_fegsage_ca_rbitlog = to_torch_cgnn
    collate_fn_fegsage_ca_rbitlog = collate_fn_cgnn
    to_torch_fegsage_ca_rbitlogsep = to_torch_cgnn
    collate_fn_fegsage_ca_rbitlogsep = collate_fn_cgnn
    to_torch_fegsage_ca_rbitlogsepgate = to_torch_cgnn
    collate_fn_fegsage_ca_rbitlogsepgate = collate_fn_cgnn
    to_torch_fegsage_ca_rbid = to_torch_cgnn
    collate_fn_fegsage_ca_rbid = collate_fn_cgnn
    to_torch_fegsage_ca_rbi_wol = to_torch_cgnn
    collate_fn_fegsage_ca_rbi_wol = collate_fn_cgnn
    to_torch_fegsage_ca_rbi_woe = to_torch_cgnn
    collate_fn_fegsage_ca_rbi_woe = collate_fn_cgnn
    to_torch_fegsage_ca_rbi_na = to_torch_cgnn
    collate_fn_fegsage_ca_rbi_na = collate_fn_cgnn
    to_torch_fegsage_ca_rbil = to_torch_cgnn
    collate_fn_fegsage_ca_rbil = collate_fn_cgnn
    to_torch_fegsage_ca_rbim = to_torch_cgnn
    collate_fn_fegsage_ca_rbim = collate_fn_cgnn
    to_torch_fegsage_ca_rbiq = to_torch_cgnn
    collate_fn_fegsage_ca_rbiq = collate_fn_cgnn
    to_torch_fegsage_ca_rbiw = to_torch_cgnn
    collate_fn_fegsage_ca_rbiw = collate_fn_cgnn
    to_torch_fegsage_ca_rbiw_wol = to_torch_cgnn
    collate_fn_fegsage_ca_rbiw_wol = collate_fn_cgnn
    to_torch_fegsage_ca_rbiw_wom = to_torch_cgnn
    collate_fn_fegsage_ca_rbiw_wom = collate_fn_cgnn
    to_torch_fegsage_ca_rbiw_na = to_torch_cgnn
    collate_fn_fegsage_ca_rbiw_na = collate_fn_cgnn
    to_torch_fegsage_ca_rbiw2 = to_torch_cgnn
    collate_fn_fegsage_ca_rbiw2 = collate_fn_cgnn
    to_torch_fegsage_ca_rbiw3 = to_torch_cgnn
    collate_fn_fegsage_ca_rbiw3 = collate_fn_cgnn


    def to_torch_transformer_stage(self, df_known, df_auxil, target_row):
        df_known = self.norm_fcol(df_known)
        df_auxil = self.norm_fcol(df_auxil)
        target_row = self.norm_fcol(target_row)
        
        q_loc_df = target_row[list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys())].copy()
        q_serie = torch.from_numpy(q_loc_df.values).float()

        answer = torch.from_numpy(target_row[['Result_norm']].values).float()

        df_known = df_known[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
        known_serie = torch.from_numpy(encode_and_bind(df_known, 'op', self.dataset_info["op_dic"]).values).float()
        # known_serie[:, 1+len(list(self.dataset_info["non_eu_col"].keys())):] -= q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):]

        df_auxil = df_auxil[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
        auxil_serie = torch.from_numpy(encode_and_bind(df_auxil, 'op', self.dataset_info["op_dic"]).values).float()
        # auxil_serie[:, (1 + len(list(self.dataset_info["non_eu_col"].keys()))) : (1 + len(list(self.dataset_info["non_eu_col"].keys())) + len(list(self.dataset_info["eu_col"].keys())))] -= q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):]
        
        return q_serie, known_serie, auxil_serie, answer

    
    def collate_fn_transformer_stage(self, examples):
        q_series = torch.concat([ex[0] for ex in examples], 0)
        known_lenths = torch.tensor([len(ex[1]) for ex in examples])
        auxil_lenths = torch.tensor([len(ex[2]) for ex in examples])
        input_series = pad_sequence([torch.cat([ex[1], ex[2]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)
        answers = torch.tensor([ex[3] for ex in examples])

        return q_series, known_lenths, auxil_lenths, input_series, answers

    
    to_torch_transformer_na2 = to_torch_transformer_stage
    collate_fn_transformer_na2 = collate_fn_transformer_stage
    to_torch_transformer_d2n = to_torch_transformer_stage
    collate_fn_transformer_d2n = collate_fn_transformer_stage
    to_torch_transformer_dt3n = to_torch_transformer_stage
    collate_fn_transformer_dt3n = collate_fn_transformer_stage


    def to_torch_transformer_idw(self, df_known, df_auxil, target_row):
        df_known = self.norm_fcol(df_known)
        df_auxil = self.norm_fcol(df_auxil)
        target_row = self.norm_fcol(target_row)

        # a_token = target_row[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
        # a_serie = torch.from_numpy(encode_and_bind(a_token, 'op', self.dataset_info["op_dic"]).values).float()

        q_loc_df = target_row[list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys())].copy()
        q_serie = torch.from_numpy(q_loc_df.values).float()

        answer = torch.from_numpy(target_row[['Result_norm']].values).float()

        df_known = df_known[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
        known_serie = torch.from_numpy(encode_and_bind(df_known, 'op', self.dataset_info["op_dic"]).values).float()   
        rc_k = known_serie[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] - q_serie[:, (len(list(self.dataset_info["non_eu_col"].keys()))):(len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))]
        known_serie_c = known_serie.clone()
        known_serie_c[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] = rc_k
        # known_serie[:, 1+len(list(self.dataset_info["non_eu_col"].keys())):] -= q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):]

        df_auxil = df_auxil[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
        auxil_serie = torch.from_numpy(encode_and_bind(df_auxil, 'op', self.dataset_info["op_dic"]).values).float()
        # auxil_serie[:, (1 + len(list(self.dataset_info["non_eu_col"].keys()))) : (1 + len(list(self.dataset_info["non_eu_col"].keys())) + len(list(self.dataset_info["eu_col"].keys())))] -= q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):]
        rc_a = auxil_serie[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] - q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys()))]
        auxil_serie_c = auxil_serie.clone()
        auxil_serie_c[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] = rc_a

        c = df_known[list(self.dataset_info["eu_col"].keys())].values
        v = df_known['Result_norm'].values
        ci = q_loc_df[list(self.dataset_info["eu_col"].keys())].values
        idw_np = self.idw_interpolation(c, v, ci)
        idw = torch.from_numpy(idw_np).float()
        
        return q_serie, known_serie, auxil_serie, known_serie_c, auxil_serie_c, answer, idw

    
    def collate_fn_transformer_idw(self, examples):
        q_series = torch.concat([ex[0] for ex in examples], 0)
        known_lenths = torch.tensor([len(ex[1]) for ex in examples])
        auxil_lenths = torch.tensor([len(ex[2]) for ex in examples])
        input_series = pad_sequence([torch.cat([ex[1], ex[2]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)
        input_series_c = pad_sequence([torch.cat([ex[3], ex[4]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)
        # known_series = pad_sequence([ex[1] for ex in examples], batch_first=True, padding_value=0.0)
        answers = torch.tensor([ex[5] for ex in examples])
        idws = torch.tensor([ex[6] for ex in examples])
        # a_series = pad_sequence([torch.cat([ex[7], ex[1], ex[2]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)

        # return q_series, known_lenths, auxil_lenths, input_series, known_series, answers, idws
        return q_series, known_lenths, auxil_lenths, input_series, input_series_c, answers, idws


    to_torch_transformer_dt3n2 = to_torch_transformer_idw
    collate_fn_transformer_dt3n2 = collate_fn_transformer_idw
    to_torch_transformer_dt3n3 = to_torch_transformer_idw
    collate_fn_transformer_dt3n3 = collate_fn_transformer_idw
    to_torch_transformer_dt3n4 = to_torch_transformer_idw
    collate_fn_transformer_dt3n4 = collate_fn_transformer_idw
    to_torch_transformer_dt3an = to_torch_transformer_idw
    collate_fn_transformer_dt3an = collate_fn_transformer_idw


    def to_torch_trsage(self, df_known, df_auxil, target_row):
        df_known = self.norm_fcol(df_known)
        df_auxil = self.norm_fcol(df_auxil)
        target_row = self.norm_fcol(target_row)

        # a_token = target_row[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
        # a_serie = torch.from_numpy(encode_and_bind(a_token, 'op', self.dataset_info["op_dic"]).values).float()

        q_loc_df = target_row[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]].copy()
        q_serie = torch.from_numpy(encode_and_bind(q_loc_df, 'op', self.dataset_info["op_dic"]).values).float()
        # print(q_serie.size())
        q_serie[:, 0] = 0

        answer = torch.from_numpy(target_row[['Result_norm']].values).float()

        df_known = df_known[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
        known_serie = torch.from_numpy(encode_and_bind(df_known, 'op', self.dataset_info["op_dic"]).values).float()   
        # rc_k = known_serie[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] - q_serie[:, (len(list(self.dataset_info["non_eu_col"].keys()))):(len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))]
        # known_serie_c = known_serie.clone()
        # known_serie_c[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] = rc_k
        # known_serie[:, 1+len(list(self.dataset_info["non_eu_col"].keys())):] -= q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):]

        df_auxil = df_auxil[['Result_norm', ] + list(self.dataset_info["non_eu_col"].keys()) + list(self.dataset_info["eu_col"].keys()) + ['op', ]]
        auxil_serie = torch.from_numpy(encode_and_bind(df_auxil, 'op', self.dataset_info["op_dic"]).values).float()
        # auxil_serie[:, (1 + len(list(self.dataset_info["non_eu_col"].keys()))) : (1 + len(list(self.dataset_info["non_eu_col"].keys())) + len(list(self.dataset_info["eu_col"].keys())))] -= q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):]
        # rc_a = auxil_serie[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] - q_serie[:, len(list(self.dataset_info["non_eu_col"].keys())):len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys()))]
        # auxil_serie_c = auxil_serie.clone()
        # auxil_serie_c[:, (1+len(list(self.dataset_info["non_eu_col"].keys()))):(1+len(list(self.dataset_info["non_eu_col"].keys()))+len(list(self.dataset_info["eu_col"].keys())))] = rc_a

        # c = df_known[list(self.dataset_info["eu_col"].keys())].values
        # v = df_known['Result_norm'].values
        # ci = q_loc_df[list(self.dataset_info["eu_col"].keys())].values
        # idw_np = self.idw_interpolation(c, v, ci)
        # idw = torch.from_numpy(idw_np).float()
        
        # return q_serie, known_serie, auxil_serie, known_serie_c, auxil_serie_c, answer, idw
        return q_serie, known_serie, auxil_serie, answer

    
    def collate_fn_trsage(self, examples):
        q_series = torch.concat([ex[0] for ex in examples], 0)
        known_lenths = torch.tensor([len(ex[1]) for ex in examples])
        auxil_lenths = torch.tensor([len(ex[2]) for ex in examples])
        input_series = pad_sequence([torch.cat([ex[1], ex[2]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)
        # input_series_c = pad_sequence([torch.cat([ex[3], ex[4]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)
        # known_series = pad_sequence([ex[1] for ex in examples], batch_first=True, padding_value=0.0)
        # answers = torch.tensor([ex[5] for ex in examples])
        answers = torch.tensor([ex[3] for ex in examples])
        # idws = torch.tensor([ex[6] for ex in examples])
        # a_series = pad_sequence([torch.cat([ex[7], ex[1], ex[2]], dim=0) for ex in examples], batch_first=True, padding_value=0.0)

        # return q_series, known_lenths, auxil_lenths, input_series, known_series, answers, idws
        return q_series, known_lenths, auxil_lenths, input_series, answers


    to_torch_trsage2 = to_torch_trsage
    collate_fn_trsage2 = collate_fn_trsage
    to_torch_trsage3 = to_torch_trsage
    collate_fn_trsage3 = collate_fn_trsage
    to_torch_trsage4 = to_torch_trsage
    collate_fn_trsage4 = collate_fn_trsage
    to_torch_trsage5 = to_torch_trsage
    collate_fn_trsage5 = collate_fn_trsage
    to_torch_trsage5_na = to_torch_trsage
    collate_fn_trsage5_na = collate_fn_trsage
    to_torch_trsage5_klrs = to_torch_trsage
    collate_fn_trsage5_klrs = collate_fn_trsage
    to_torch_trsage5_nkl = to_torch_trsage
    collate_fn_trsage5_nkl = collate_fn_trsage
    to_torch_trsage5_nd = to_torch_trsage
    collate_fn_trsage5_nd = collate_fn_trsage
    to_torch_transformer_na = to_torch_trsage
    collate_fn_transformer_na = collate_fn_trsage
    to_torch_trsage5_gee = to_torch_trsage
    collate_fn_trsage5_gee = collate_fn_trsage
    to_torch_trsage5_rb = to_torch_trsage
    collate_fn_trsage5_rb = collate_fn_trsage


    def to_torch_ssin(self, df_known, df_auxil, target_row):
        df_known = self.norm_fcol(df_known)
        df_auxil = self.norm_fcol(df_auxil)
        target_row = self.norm_fcol(target_row)

        graph_candidates = pd.concat([target_row, df_known], axis=0, ignore_index=True)
        
        coords = torch.from_numpy(graph_candidates[list(self.dataset_info["eu_col"].keys())].values).float()
        # coords = coords - coords[0, :]
        
        answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()

        full_df = pd.concat([df_known, df_auxil], axis=0, ignore_index=True)
        aggregated_df = full_df.groupby(list(self.dataset_info["eu_col"].keys()) + ['op']).mean().reset_index()
        features = torch.zeros((len(graph_candidates), len(self.dataset_info["op_dic"]) + 1))
        possible_values = list(self.dataset_info["op_dic"].keys())
        possible_values.sort()
        for op in possible_values:
            aggregated_df_op = aggregated_df[aggregated_df['op']==op]
            interpolated_grid = torch.zeros((len(graph_candidates), 1))
            if len(aggregated_df_op) != 0:
                c = aggregated_df_op[list(self.dataset_info["eu_col"].keys())].values
                v = aggregated_df_op['Result_norm'].values
                ci = graph_candidates[list(self.dataset_info["eu_col"].keys())].values
                interpolated_values = self.idw_interpolation(c, v, ci)
                interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(graph_candidates), 1))
            features[:, self.dataset_info["op_dic"][op]:self.dataset_info["op_dic"][op]+1] = interpolated_grid
        
        conditions = graph_candidates['op'] == self.dataset_info["tgt_op"]
        features[:, -1] = torch.from_numpy(np.where(conditions, graph_candidates['Thing'], (self.dataset_info['lowest_rank']-1)/self.dataset_info["non_eu_col"]["Thing"][1]))
        features = features.float()

        features[0, :] = 0

        return features, coords, answers


    def collate_fn_ssin(self, examples):
        input_lenths = torch.tensor([len(ex[0]) for ex in examples if len(ex[0]) > 2])
        x_b = pad_sequence([ex[0] for ex in examples if len(ex[0]) > 2], batch_first=True, padding_value=0.0)
        c_b = pad_sequence([ex[1] for ex in examples if len(ex[1]) > 2], batch_first=True, padding_value=0.0)
        y_b = pad_sequence([ex[2] for ex in examples if len(ex[2]) > 2], batch_first=True, padding_value=0.0)
        
        return x_b, c_b, y_b, input_lenths
