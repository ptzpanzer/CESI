import os
import json
import shutil
import pickle
import concurrent.futures
import pandas as pd
import numpy as np


def build_folder_and_clean(path):
    check = os.path.exists(path)
    if check:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def norm(d):
    d_list = []
    for op in d['op'].unique():
        d_op = d[d['op']==op].copy()
        if op in dataset_info["tgt_logical"]:
            op_norm = dataset_info["tgt_op"]
        else:
            op_norm = op

        d_op['Result_norm'] = (d_op['Result'] - dic_op_minmax[op_norm][0]) / (dic_op_minmax[op_norm][1] - dic_op_minmax[op_norm][0])

        # poison dataset
        random_choices = np.random.rand(len(d_op))
        d_op['Result_poison'] = d_op['Result_norm']
        mask_50 = random_choices < 0.5
        d_op.loc[mask_50, 'Result_poison'] += np.random.normal(0, 0.05, size=mask_50.sum())
        mask_20_1 = (random_choices >= 0.5) & (random_choices < 0.7)
        d_op.loc[mask_20_1, 'Result_poison'] += np.random.normal(0, 0.02, size=mask_20_1.sum())
        mask_20_2 = (random_choices >= 0.7) & (random_choices < 0.9)
        d_op.loc[mask_20_2, 'Result_poison'] += np.random.normal(0, 0.07, size=mask_20_2.sum())
        mask_10 = random_choices >= 0.9
        
        d_list.append(d_op)
    return pd.concat(d_list, axis=0, ignore_index=False).drop(columns=['Result'])


def process_child(t):
    file, out_path, burn_rate = t

    file_fullpath = origin_path + 'Dataset_Separation/' + file
    df = pd.read_csv(file_fullpath, sep=';')
    # drop everything in bad quality
    df = df[df['Thing']>=dataset_info['lowest_rank']]
    # drop everything with non-whitelisted op
    op_whitelist = list(dataset_info["op_dic"].keys())
    for holdout in dataset_info["holdouts"].keys():
        op_whitelist = op_whitelist + dataset_info["holdouts"][holdout]["train"] + dataset_info["holdouts"][holdout]["test"] + dataset_info["holdouts"][holdout]["eval"]
    op_whitelist = list(set(op_whitelist))
    df = df[df['op'].isin(op_whitelist)]

    # burn dataset, burn df_auxil only
    list_auxil = list(dataset_info["op_dic"].keys())
    list_auxil.remove(dataset_info["tgt_op"])
    df_auxil = df[df['op'].isin(list_auxil)]
    num_rows_to_remove = int(len(df_auxil) * burn_rate)
    rows_to_remove = df_auxil.sample(n=num_rows_to_remove).index
    df = df.drop(rows_to_remove)

    # norm and poison dataset
    df = norm(d=df)

    print(f'\t{file_fullpath} Done!', end="\r", flush=True)
    return out_path + 'Dataset_Separation/' + file, df
    

origin_path = f"./OceanAt_res250_reg4c/"
with open(origin_path + f'meta_data.json', 'r') as f:
    dataset_info = json.load(f)
with open(origin_path + f"Folds_Info/norm_4_0.json", 'r') as f:
    dic_op_minmax = json.load(f)

list_csv = []
for h in range(4):
    with open(origin_path + f"Folds_Info/divide_set_4_{h}.info", 'rb') as f:
        divide_set = pickle.load(f)
    list_csv = list_csv + divide_set[0] + divide_set[1] + divide_set[2]
    print(f'Holdout {h} combined!')
list_csv = list(set(list_csv))
    
orders = [0.2, 0.4, 0.6, 0.8]
for order in orders:
    print(f'Starting order {order}!')
    
    out_path = f"./OceanAt_res250_reg4c_b_{order}/"
    build_folder_and_clean(out_path + 'Dataset_Separation/')

    scene_list = [(file, out_path, order) for file in list_csv]
    total_df_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_child, file_name) for file_name in scene_list]
        for future in concurrent.futures.as_completed(futures):
            file_name, file_content = future.result()
            total_df_dict[file_name] = file_content

    for key in total_df_dict.keys():
        total_df_dict[key].to_csv(key, header=True, index=False, sep=';')

    print(f'order {order} done!')