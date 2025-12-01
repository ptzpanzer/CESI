# this script convert raw meteo data from 7 meteo stations into SAQN-Styled data
#   - Separation:
#       - Files in ./Meteo202205_Original/
#   - Output:
#       - Filtered valid data to ./Meteo202205_Decompressed/

from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from dateutil import tz
import pandas as pd
import pytz


# ===================== FUNCTIONS ===============================#
# This is used to generate time slices for 1 years within the interested zone
def time_slices(start_year, end_year):
    time_start = datetime(start_year, 1, 1, 0, 0, 0, tzinfo=tz.tzutc())
    time_end = datetime(end_year, 1, 1, 0, 0, 0, tzinfo=tz.tzutc())

    time_lst = []
    while time_start < time_end:
        t1 = time_start.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        t2 = (time_start + relativedelta(years=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        time_lst.append([t1, t2])
        time_start += relativedelta(years=1)
    return time_lst


# This is used to convert CET-TimeString in onoca stations to UTC ones
def convert_time_onoca(x):
    x_dt = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    x_dt = pytz.timezone('CET').localize(x_dt)
    rtn = x_dt.astimezone(pytz.timezone('UTC')).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    return rtn


# This is used to convert CET/CEST/UTC-TimeString in uni station to UTC ones
def convert_time_uni(x):
    x_dt = datetime.strptime(x, '%Y:%m:%d:%H:%M:%S')

    x_day = datetime(year=x_dt.year, month=x_dt.month, day=x_dt.day)
    # After this day, timestamp of uni changed to UTC
    decision_day = datetime(2019, 11, 13)
    if x_day <= decision_day:
        x_dt = pytz.timezone('Europe/Berlin').localize(x_dt)
    else:
        x_dt = pytz.timezone('UTC').localize(x_dt)

    rtn = x_dt.astimezone(pytz.timezone('UTC')).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    return rtn


# This is used to convert CET/CEST-TimeString in uni2 station to UTC ones
def convert_time_uni2(x):
    x_dt = datetime.strptime(x, '%Y:%m:%d:%H:%M:%S')
    x_dt = pytz.timezone('Europe/Berlin').localize(x_dt)
    rtn = x_dt.astimezone(pytz.timezone('UTC')).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    return rtn
# ===================== FUNCTIONS ===============================#


# ===================== SETTINGS ================================#
def raw_standarizer(origin_path, output_path, year_start, year_end):
    time_list = time_slices(year_start, year_end)
    folders = os.listdir(origin_path)
    onoca = ['kobelweg', 'koenigsplatz', 'rotes_tor', 'stephan', 'zentralklinikum']
    locations = {
        'kobelweg': [10.82027, 48.38166, 535.0],
        'koenigsplatz': [10.89416, 48.36556, 496.0],
        'rotes_tor': [10.90639, 48.35749, 478.0],
        'stephan': [10.900002, 48.37444, 479.0],
        'zentralklinikum': [10.83805, 48.38388, 537.0],
        'uni': [10.8971, 48.33472, 513.0],
        'uni2': [10.8971, 48.33472, 513.0]
    }
    ops = ['TE', 'FE', 'DR', 'RD', 'WV', 'WD', 'SO']
    op_proj = {'TE': 'ta', 'FE': 'hur', 'DR': 'plev', 'RD': 'precip', 'WV': 'wdir', 'WD': 'wspeed', 'SO': 'globalrad'}
    for window_start, window_end in time_list:
        print(f'Working on Time Window {window_start} to {window_end}')
        dic_stations = {}
        dic_out = {}
        for folder in folders:
            files = os.listdir(origin_path + folder + '/')
            for file in files:
                print('\tChecking on /' + folder + '/' + file)
                if folder in onoca:
                    df = pd.read_csv(origin_path + folder + '/' + file, sep=',')
                    df['timestamp'] = df['timestamp'].apply(convert_time_onoca)
                elif folder == 'uni':
                    if '_dat' in file:
                        df = pd.read_csv(origin_path + folder + '/' + file, sep=' ')
                        if len(df.columns) == 20:
                            df.columns = ["timestamp", "TE", "SO", "DR", "WR", "FE", "RE", "RD", "WG", "WS", "WD", "WC",
                                          "WV", "ZA", "ZB", "ZC", "ZD", "ZE", "TK", "UH"]
                        elif len(df.columns) == 16:
                            df.columns = ["timestamp", "TE", "SO", "DR", "WR", "FE", "RE", "RD", "WG", "WS", "WD", "WC",
                                          "WV", "ZA", "ZB", "ZC"]
                        df['timestamp'] = df['timestamp'].apply(convert_time_uni)
                    else:
                        print('\t\tThis is not a _dat file')
                        continue
                elif folder == 'uni2':
                    if '_dat' in file:
                        df = pd.read_csv(origin_path + folder + '/' + file, sep=' ')
                        df.columns = ["timestamp", "TE", "FE", "TD", "DR", "WR", "WV", "WG", "WS", "WD", "RE", "RD", "SO",
                                      "UB", "FS", "DB", "TP", "ZR", "ZW", "ZG", "ZP", "ZX", "ZH", "UH"]
                        df['timestamp'] = df['timestamp'].apply(convert_time_uni2)
                    else:
                        print('\t\tThis is not a _dat file')
                        continue

                df_windowed = df[(df['timestamp'] >= window_start) & (df['timestamp'] < window_end)]
                if folder not in dic_stations.keys():
                    dic_stations[folder] = pd.DataFrame()
                dic_stations[folder] = pd.concat([dic_stations[folder], df_windowed])

            print(f'\n\tFor {folder} we have {len(dic_stations[folder])} Observations.')
            dic_stations[folder] = dic_stations[folder].drop_duplicates()
            print(f'\tAfter deduplicate we have {len(dic_stations[folder])} Observations.')

            for op in ops:
                if op in dic_stations[folder].columns:
                    print(f'\t\tGenerating Table for {op}.')
                    df_tmp = dic_stations[folder][['timestamp', op]]
                    df_tmp['Time_End'] = df_tmp['timestamp']
                    df_tmp.insert(0, 'Thing', folder)
                    df_tmp.insert(0, 'Datastream', folder + ':' + op)
                    df_tmp.insert(0, 'Longitude', locations[folder][0])
                    df_tmp.insert(0, 'Latitude', locations[folder][1])
                    df_tmp.insert(0, 'Altitude', locations[folder][2])
                    df_tmp = df_tmp[['Thing', 'Datastream', 'timestamp', 'Time_End', 'Longitude', 'Latitude', 'Altitude', op]]
                    df_tmp.columns = ['Thing', 'Datastream', 'Time_Start', 'Time_End', 'Longitude', 'Latitude', 'Altitude', 'Result']
                    if op not in dic_out.keys():
                        dic_out[op] = pd.DataFrame(columns=['Thing', 'Datastream', 'Time_Start', 'Time_End', 'Longitude', 'Latitude', 'Altitude', 'Result'])
                    dic_out[op] = pd.concat([dic_out[op], df_tmp])
                else:
                    print(f'\t\t{op} Not found in station {folder}.')

            print(f'\t\tRemapping on folder {folder} finished.')

        print(f'\n\tStarting output for {window_start} - {window_end}')
        for key in dic_out.keys():
            dic_out[key].to_csv(output_path + f'{op_proj[key]}-{window_start[0:4]}.csv',
                                sep=';', header=True, mode='w', index=False)
# ===================== SETTINGS ================================#
