# then divide and analyse datasets as 5-fold
import os
import pickle
import pandas as pd
import json


# build up the given path
def build_folder_and_clean(path):
    check = os.path.exists(path)
    if check:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(path)


def get_set(lists, indexs):
    rtn = []
    for index in indexs:
        rtn += lists[index]
    return rtn


process_list = [
    "ABO_res250_reg4c", 'OceanAt_res250_reg4c', 'SAQN_res250_reg4c',
]

for item in process_list:
    print(f"\nProcessing {item}:")
    in_path = f'./{item}/Dataset_Separation/'
    out_path = f'./{item}/Folds_Info/'
    build_folder_and_clean(out_path)

    files = [file for file in os.listdir(in_path) if file.endswith(".csv")]
    len_all = len(files)
    files.sort()

    if len_all > 20000:
        files = files[-20000:]
    
    print(f"Total files: {len_all}, cut down to: {len(files)}")
    
    fold_count = 5
    fold_len = len(files) // fold_count
    folds = [files[i:i+fold_len] for i in range(0, len(files), fold_len)]

    with open(f'./{item}/meta_data.json', 'r') as json_file:
        settings = json.load(json_file)
    
    for i in range(4, 5):
        train_index = [(i-2)%5, (i-3)%5, (i-4)%5]
        test_index = [(i-1)%5]
        eval_index = [i%5]
        train_set = get_set(folds, train_index)
        test_set = get_set(folds, test_index)
        eval_set = get_set(folds, eval_index)
        
        holdouts_list = list(settings["holdouts"].keys())
        holdouts_list.sort()
        for j in holdouts_list:
            print(f"\n\tProcessing fold {i}, houldout {j}:")

            # save set divide info
            with open(out_path + f'divide_set_{i}_{j}.info', 'wb') as f:
                pickle.dump([train_set, test_set, eval_set], f)

            # work on normalization, noticing:
            #     - train set only contains readings from non-official stations on 60% of time slices
            #     - normalization info should be only analyzed within these data
            dic_op_df = {}
            #         - only on 60% of time slices
            for idx, file in enumerate(train_set):
                print(f'\t\tWorking on fold {i}, houldout {j}, file {idx}/{len(train_set)}', end="\r", flush=True)
                df = pd.read_csv(in_path + file, sep=';')
                for op in df['op'].unique():
                    if (op not in settings["holdouts"][j]["test"] and op not in settings["holdouts"][j]["eval"]) or (op in settings["holdouts"][j]["train"]):
                        df_op = df[df['op']==op]
                        if op in settings["tgt_logical"]:
                            real_op = settings["tgt_op"]
                        else:
                            real_op = op
                        if real_op not in dic_op_df.keys():
                            dic_op_df[real_op] = [df_op]
                        else:
                            dic_op_df[real_op].append(df_op)

                        # if real_op == "mcpm10" and df_op['Result'].max() > 80:
                        #     print(file)
                        #     print(df)

            for key in dic_op_df.keys():
                dic_op_df[key] = pd.concat(dic_op_df[key])
            
            # calculate mean and std then dump result
            dic_op_minmax = {}
            for op in dic_op_df.keys():
                op_max = dic_op_df[op]['Result'].max()
                op_min = dic_op_df[op]['Result'].min()
                op_mean = dic_op_df[op]['Result'].mean()
                op_std = dic_op_df[op]['Result'].std()
                dic_op_minmax[op] = [op_min, op_max, op_mean, op_std]
            with open(out_path + f'norm_{i}_{j}.json', 'w') as f:
                json.dump(dic_op_minmax, f)
    