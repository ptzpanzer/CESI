# Filter interested data from SAQN dataset (without 7 METO Stations)
#   - Separation:
#       - Files in ./SAQN2021_CSV_Decompressed/
#       - ./Boundary_Analyzer_Settings_SAQN.txt
#   - Output:
#       - Filtered interested data to ./SAQN2021_CSV_Interested/
#       - Data value Analysis file to ./SAQN2021_Analysis_DataValue/

import os
import numpy as np
import pandas as pd


def boundary_analyzer_0(csv_path, dic_settings, csv_output_path, analysis_output_path):
    files = os.listdir(csv_path)

    for file in files:
        print(f'Working on {file}:')
        op = file.split('-')[0]
        if op not in dic_settings['list_of_ops']:
            print('\tWe do not care this.')
        else:
            df = pd.DataFrame()
            with pd.read_csv(csv_path + file, sep=';', chunksize=10 ** 6) as reader:
                for chunk in reader:
                    print('\tStarting new Chunk.')
                    filtered_chunk = chunk[chunk['Thing'].isin(dic_settings['list_of_things'])]
                    size = len(filtered_chunk)
                    print(f'\t\tOriginal Chunk Size: {size}')
                    # Drop Null-Value
                    filtered_chunk['Result'] = filtered_chunk['Result'].replace(to_replace='None', value=np.nan)
                    filtered_chunk = filtered_chunk[filtered_chunk['Result'].notna()]
                    size = len(filtered_chunk)
                    print(f'\t\tChunk Size after Drop Null-Value: {size}')
                    # Drop Non-Numeric Value
                    filtered_chunk = filtered_chunk[pd.to_numeric(filtered_chunk['Result'], errors='coerce').notnull()]
                    size = len(filtered_chunk)
                    print(f'\t\tChunk Size after Drop Non-Numeric Value: {size}')
                    # Take Operations
                    for command in dic_settings['operations']:
                        if command[1] == 'DELT':
                            filtered_chunk = filtered_chunk[filtered_chunk['Datastream'] != command[2]]
                            size = len(filtered_chunk)
                            print(f'\t\tChunk Size after Take Operations: {size}')
                        elif command[1] == 'UNIT':
                            filtered_chunk.Result = filtered_chunk.Result.astype(float)
                            filtered_chunk.loc[filtered_chunk.Datastream == command[2], 'Result'] *= float(command[3])
                            size = len(filtered_chunk)
                            print(f'\t\tChunk Size after Take Operations: {size}')
                    df = pd.concat([df, filtered_chunk])
            if len(df.index) != 0:
                output_name = csv_output_path + file
                df.to_csv(output_name, sep=';', header=True, mode='w', index=False)
                output_name = analysis_output_path + file[:-4] + ' - description.csv'
                df['Result'] = df['Result'].map(float)
                df_count = df['Result'].value_counts()
                df_count = df_count.sort_index(ascending=True)
                df_count.to_csv(output_name, sep=';')
