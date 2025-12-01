# filter Interested data with reasonable bound, set as intermedia format (SAQN_CSV_format)
#   - Separation:
#       - Files in ./SAQN2021_CSV_Interested/
#       - ./Boundary_Analyzer_Settings_SAQN.txt
#       - ./SAQN2021_DataValue_Combined/Data_bounds.txt
#   - Output:
#       - Filtered valid data to ./SAQN2021_CSV_valid/

import os
import pandas as pd
from dateutil.relativedelta import relativedelta
import dateutil.parser


def drop_impossible(dd, bound):
    dd = dd[(dd['Result'] >= bound[0]) & (dd['Result'] <= bound[1])]
    return dd


# This is used to generate time slices in resolution h within interested time span
def locate_time_slice(row):
    time = dateutil.parser.isoparse(row['Time_Start'])
    time_slice = time.replace(minute=0, second=0)
    t1 = time_slice.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    return t1


def data_filter(csv_interested_path, csv_file, dic_bound):
    op = csv_file.split('-')[0]
    print(f'Working on {csv_interested_path}{csv_file}, op is {op}:')
    d = pd.read_csv(csv_interested_path + csv_file, sep=';')
    d['Result'] = d['Result'].map(float)
    print(f'\tOriginal Chunk Size:      {len(d)}')
    # Drop Impossible Value
    d = drop_impossible(d, dic_bound[op])
    d['op'] = op
    d['Time'] = d.apply(locate_time_slice, axis = 1)
    d = d[['op', 'Thing', 'Datastream', 'Time', 'Time_Start', 'Time_End', 'Longitude', 'Latitude', 'Altitude', 'Result']]
    print(f'\tAfter Dropped Chunk Size: {len(d)}')
    return d


def multi_print(x):
    d, output_path, time = x
    filtered_chunk = d[d['Time'] == time]
    output_name = time.replace(':', '-') + '.csv'
    if os.path.exists(output_path + output_name):
        filtered_chunk.to_csv(output_path + output_name, sep=';', header=False, mode='a', index=False)
    else:
        filtered_chunk.to_csv(output_path + output_name, sep=';', header=True, mode='a', index=False)
