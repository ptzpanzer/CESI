import os
import json
import pickle
import random
import shutil
import pandas as pd


def get_region(longitude, latitude):
    x_region = int(longitude / 250 * 2)
    y_region = int(latitude / 250 * 2)
    if x_region < 0 or x_region > 1 or y_region < 0 or y_region > 1:
        return None
    region = y_region * 2 + x_region
    return region


nt_datasets = ["ABO_res250", 'OceanAt_res250', 'SAQN_res250']

for ds in nt_datasets:
    src_path = f"./{ds}/"
    csv_files = os.listdir(src_path + "Dataset_Separation/")
    csv_files.sort()
    with open(src_path + f'meta_data.json', 'r') as f:
        dataset_info = json.load(f)
    targets = dataset_info["tgt_logical"]
    targets.append(dataset_info["tgt_op"])

    print(f"ds: {ds}, targets: {targets}")

    tgt_path = f"./{ds}_reg4c/"
    check = os.path.exists(tgt_path)
    if check:
        shutil.rmtree(tgt_path)
        os.makedirs(tgt_path)
    else:
        os.makedirs(tgt_path)
    os.makedirs(tgt_path + "Dataset_Separation/")

    num_regions = 4
    region_sets = {i: {'train': set(), 'eval': set(), 'test': set(), "files": []} for i in range(num_regions)}
    for i in range(num_regions):
        region_sets[i]['eval'].add(i)

        other_regions = set(range(num_regions)) - region_sets[i]['eval']
        region_sets[i]['train'].update(other_regions)
        region_sets[i]['test'].update(other_regions)

    with open(tgt_path + "log.pkl", 'wb') as file:
        pickle.dump(region_sets, file)

    for file in csv_files:
        if ".csv" not in file:
            continue

        df = pd.read_csv(src_path + "Dataset_Separation/" + file, sep=";")
        df = df[(df['Longitude'] >= 0) & (df['Longitude'] < 250) & (df['Latitude'] >= 0) & (df['Latitude'] < 250)]

        if ds in ["SAQN_res250", "LUFT_res250", ]:
            condition = ((df['op'].isin(targets + ['mcpm2p5'])) & (df['Result'] > 80))
            fdf = df[condition]
            df = df.drop(fdf.index)

        tgt_rows = df[df['op'].isin(targets)]
        for index, row in tgt_rows.iterrows():
            region = get_region(row['Longitude'], row['Latitude'])
            if region is None:
                print("None detected")
            new_op = f"{dataset_info['tgt_op']}_{region}"
            df.at[index, 'op'] = new_op

        dic_region_count = {}
        for region in range(num_regions):
            dic_region_count[region] = df[df['op'] == f"{dataset_info['tgt_op']}_{region}"].shape[0]
        dic_region_count["aux"] = df[~df['op'].str.contains(f"{dataset_info['tgt_op']}_")].shape[0]

        keep = True
        for holdout in range(num_regions):
            train_count = 0
            for train in region_sets[holdout]["train"]:
                train_count += dic_region_count[train]

            test_count = 0
            for test in region_sets[holdout]["test"]:
                test_count += dic_region_count[test]

            eval_count = 0
            for evaluation in region_sets[holdout]["eval"]:
                eval_count += dic_region_count[evaluation]

            aux_count = dic_region_count["aux"]

            if train_count < 5 or test_count < 1 or eval_count < 1 or aux_count < 10:
                keep = False
                print(
                    f'False {file} on holdout {holdout}: train_count {train_count}, test_count {test_count}, eval_count {eval_count}, aux_count {aux_count}')
                break

        if keep:
            df.to_csv(tgt_path + "Dataset_Separation/" + file, header=True, index=False, sep=';')

    print(f"{ds} done.")