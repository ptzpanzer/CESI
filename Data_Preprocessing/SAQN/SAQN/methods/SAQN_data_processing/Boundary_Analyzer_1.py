# Combine Data value Analysis file generated in the last step, Give reasonable data bounds for outlier filtering
#   - Separation:
#       - Files in ./SAQN2021_Analysis_DataValue/
#   - Output:
#       - Combined Data value Analysis file to ./SAQN2021_DataValue_Combined/
#       - Reasonable data bounds file as ./SAQN2021_DataValue_Combined/Data_bounds.txt

import os
import pandas as pd


def boundary_analyzer_1(analysis_paths, analysis_combined_output_path):
    files = []
    for path in analysis_paths:
        path_folder_list = os.listdir(path)
        for file in path_folder_list:
            files.append((path, file))

    ops = []
    for path, file in files:
        tmp = file.split('-')[0]
        if tmp not in ops:
            ops.append(tmp)

    for op in ops:
        df_op = pd.DataFrame()
        for path, file in files:
            if file.split('-')[0] == op:
                tmp = pd.read_csv(path + file, sep=';')
                df_op = pd.concat([df_op, tmp])
        if len(df_op.index) != 0:
            output_name = analysis_combined_output_path + op + ' - description.csv'
            df_op.columns = ['Result', 'Count']
            df_op['Result'] = df_op['Result'].map(float)
            df_op['Count'] = df_op['Count'].map(int)
            df_count = df_op.groupby(['Result']).sum()
            df_count = df_count.sort_values(by=['Result'], ascending=True)
            df_count.to_csv(output_name, sep=';')

    # [lower_bound, upper_bound, drop_form, drop_rate]
    lb = {
        'hur':          [0,     100,                'both-side-fixed'],
        'wdir':         [0,     360,                'both-side-fixed'],
        'precip':       [0,     100,                'both-side-fixed'],
        'mcpm2p5':      [0,     250,                'both-side-fixed'],
        'mcpm10':       [0,     250,                'both-side-fixed'],
        'ta':           [-25,   40,                 'both-side-fixed'],
        'globalrad':    [0,     float('inf'),       'left-side-fixed',  0.000],
        'wspeed':       [0,     float('inf'),       'left-side-fixed',  0.000],
        'plev':         [900,   1200,               'none-side-fixed',  0.000],
    }

    files = os.listdir(analysis_combined_output_path)

    with open(analysis_combined_output_path + 'Data_bounds.txt', 'w', encoding='utf-8') as fout:
        for file in files:
            df = pd.read_csv(analysis_combined_output_path + file, sep=';')
            df['Result'] = df['Result'].map(float)
            df['Count'] = df['Count'].map(int)

            i_start = 0
            i_end = len(df)
            op = file.split(' - ')[0]

            print(f'Calculating boundary for {op}')

            if lb[op][2] == 'both-side-fixed':
                must_higher = lb[op][0]
                must_lower = lb[op][1]
                for i in range(i_start, i_end):
                    if df['Result'][i] >= must_higher:
                        i_start = i
                        break
                for i in range(i_end - 1, i_start, -1):
                    if df['Result'][i] <= must_lower:
                        i_end = i
                        break
                upper = df['Result'][i_start]
                downer = df['Result'][i_end]
                count_down = df['Count'][0:i_start].sum()
                count_up = df['Count'][i_end:].sum()
                count_mid = df['Count'][i_start:i_end].sum()
                percent = float(count_down + count_up) / (count_down + count_mid + count_up)

                fout.write(f'/*OP*/{op}\n')
                fout.write(f'\t/*UP*/{upper}\n')
                fout.write(f'\t/*DOWN*/{downer}\n')
                fout.write(f'\t/*DPRT*/{percent}\n')
            elif lb[op][2] == 'left-side-fixed':
                must_higher = lb[op][0]
                belief_percent = lb[op][3]
                for i in range(i_start, i_end):
                    if df['Result'][i] >= must_higher:
                        i_start = i
                        break
                for i in range(i_end - 1, i_start, -1):
                    count_down = df['Count'][0:i_start].sum()
                    count_up = df['Count'][i:].sum()
                    count_mid = df['Count'][i_start:i].sum()
                    percent = float(count_down + count_up) / (count_down + count_mid + count_up)
                    if percent >= belief_percent:
                        i_end = i
                        break
                upper = df['Result'][i_start]
                downer = df['Result'][i_end]
                fout.write(f'/*OP*/{op}\n')
                fout.write(f'\t/*UP*/{upper}\n')
                fout.write(f'\t/*DOWN*/{downer}\n')
                fout.write(f'\t/*DPRT*/{percent}\n')
            elif lb[op][2] == 'none-side-fixed':
                total = 0.0
                count = 0.0
                for i in range(i_start, i_end):
                    if lb[op][0] <= df['Result'][i] <= lb[op][1]:
                        total += df['Result'][i] * df['Count'][i]
                        count += df['Count'][i]
                mean = total / count
                must_higher = lb[op][0]
                must_lower = lb[op][1]
                for i in range(i_start, i_end):
                    if df['Result'][i] >= must_higher:
                        i_start = i
                        break
                for i in range(i_end - 1, i_start, -1):
                    if df['Result'][i] <= must_lower:
                        i_end = i
                        break
                for i in range(i_start, i_end):
                    if df['Result'][i] >= mean:
                        i_mean = i
                        break
                percent = -1 * float("inf")
                routine_count = 0
                while percent < lb[op][3]:
                    if routine_count == 0:
                        routine_count = 1
                    else:
                        count_lower_mean = df['Count'][i_start:i_mean].sum()
                        count_higher_mean = df['Count'][i_mean:i_end].sum()
                        if count_lower_mean > count_higher_mean:
                            i_start += 1
                        else:
                            i_end -= 1
                    upper = df['Result'][i_start]
                    downer = df['Result'][i_end]
                    count_down = df['Count'][0:i_start].sum()
                    count_up = df['Count'][i_end:].sum()
                    count_mid = df['Count'][i_start:i_end].sum()
                    percent = float(count_down + count_up) / (count_down + count_mid + count_up)
                fout.write(f'/*OP*/{op}\n')
                fout.write(f'\t/*UP*/{upper}\n')
                fout.write(f'\t/*DOWN*/{downer}\n')
                fout.write(f'\t/*DPRT*/{percent}\n')
            else:
                print(f'What happend for {op}?')
