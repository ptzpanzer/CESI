# First collect all data from both SAQN and Meteo in the given time_slices, Then aggregate them by datastreams
#   - Separation:
#       - Files in ./SAQN2021_CSV_Valid/
#       - Files in ./Meteo202205_Valid/
#   - Output:
#       - Aggregated 1h slices in ./Dataset_Aggregated/

import os
import math
import pandas as pd
from geopy import distance
from dateutil.relativedelta import relativedelta
import dateutil.parser
from osgeo import gdal
from pyproj import Transformer


# This is used to generate time slices in resolution h within interested time span
def time_slices(time_start_str, time_end_str, resolution):
    time_start = dateutil.parser.isoparse(time_start_str)
    time_end = dateutil.parser.isoparse(time_end_str)
    time_lst = []

    time_now = time_start
    while time_now < time_end:
        t1 = time_now.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        t2 = (time_now + relativedelta(hours=resolution)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        time_lst.append([t1, t2])
        time_now += relativedelta(hours=resolution)

    return time_lst


# This is used to calculate distance from mean point
def cal_distance(df_item, ave_lon, ave_lat):
    coords_1 = (df_item['Latitude'], df_item['Longitude'])
    coords_2 = (ave_lat, ave_lon)
    return distance.distance(coords_1, coords_2).m


# This is used to transform the coordinates from (lon,lat) to GEO-TIFF pixel
def coord_trans(df_item, transformer, xOrigin, yOrigin, pixelWidth, pixelHeight):
    p1 = transformer.transform(df_item['Latitude'], df_item['Longitude'])
    df_item['Longitude'] = int((p1[0] - xOrigin) / pixelWidth)
    df_item['Latitude'] = int((yOrigin - p1[1]) / pixelHeight)
    return df_item


def child_aggregator(x):
    csv_path, csv_file, output_path = x
    geo_file = gdal.Open('./Data_Folder/CWSL_resampled.tif')
    gtf = geo_file.GetGeoTransform()
    xOrigin = gtf[0]
    yOrigin = gtf[3]
    pixelWidth = gtf[1]
    pixelHeight = -gtf[5]
    res_x = geo_file.RasterXSize
    res_y = geo_file.RasterYSize
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632")

    d = pd.read_csv(csv_path + csv_file, sep=';')

    # turn wind-speed/wind-direct to u-component/v-component
    dw = d[(d['op'] == 'wdir') | (d['op'] == 'wspeed')]
    d = d.drop(d[(d['op'] == 'wdir') | (d['op'] == 'wspeed')].index)
    time_list = list(dw['Time_Start'].unique())
    thing_list = list(dw['Thing'].unique())
    append_list = []
    for time in time_list:
        for thing in thing_list:
            filtered_chunk_ws = dw[(dw['Time_Start'] == time) & (dw['Thing'] == thing) & (dw['op'] == 'wspeed')]
            filtered_chunk_wd = dw[(dw['Time_Start'] == time) & (dw['Thing'] == thing) & (dw['op'] == 'wdir')]
            if len(filtered_chunk_ws) != 0 and len(filtered_chunk_wd) != 0:
                ws_mean = filtered_chunk_ws['Result'].mean(numeric_only=True)
                wd_mean = filtered_chunk_wd['Result'].mean(numeric_only=True)
                new_u = {
                    'op': 'wsx',
                    'Thing': thing,
                    'Datastream': thing + ':WSX',
                    'Time': filtered_chunk_ws.iloc[0]['Time'],
                    'Time_Start': filtered_chunk_ws.iloc[0]['Time_Start'],
                    'Time_End': filtered_chunk_ws.iloc[0]['Time_End'],
                    'Longitude': filtered_chunk_ws.iloc[0]['Longitude'],
                    'Latitude': filtered_chunk_ws.iloc[0]['Latitude'],
                    'Altitude': filtered_chunk_ws.iloc[0]['Altitude'],
                    'Result': ws_mean * math.sin(math.radians(wd_mean)),
                }
                new_v = {
                    'op': 'wsy',
                    'Thing': thing,
                    'Datastream': thing + ':WSY',
                    'Time': filtered_chunk_ws.iloc[0]['Time'],
                    'Time_Start': filtered_chunk_ws.iloc[0]['Time_Start'],
                    'Time_End': filtered_chunk_ws.iloc[0]['Time_End'],
                    'Longitude': filtered_chunk_ws.iloc[0]['Longitude'],
                    'Latitude': filtered_chunk_ws.iloc[0]['Latitude'],
                    'Altitude': filtered_chunk_ws.iloc[0]['Altitude'],
                    'Result': ws_mean * math.cos(math.radians(wd_mean)),
                }
                append_list += [pd.DataFrame([new_u]), pd.DataFrame([new_v])]
    d = pd.concat([d, ] + append_list, axis=0, ignore_index=True)

    op_list = list(d['op'].unique())
    thing_list = list(d['Thing'].unique())
    combined_count = 0
    for thing in thing_list:
        for op in op_list:
            print(f'Working on {csv_file}, thing: {thing}, op: {op}')
            filtered_chunk = d[(d['op'] == op) & (d['Thing'] == thing)]
            if len(filtered_chunk) != 0:
                # now filtered_chunk is 1-h slice of an op in a Thing
                # first reformat the lon and lat into grids
                df = filtered_chunk.apply(
                    coord_trans, axis=1, args=(transformer, xOrigin, yOrigin, pixelWidth, pixelHeight)
                )
                combined_count += len(df)
                # then aggregate this into grid, throw outliers away
                df_agg = df.groupby(['Longitude', 'Latitude'], as_index=False).mean(numeric_only=True)
                df_agg = df_agg[(df_agg['Longitude'] < res_x) & (df_agg['Longitude'] >= 0) &
                                (df_agg['Latitude'] < res_y) & (df_agg['Latitude'] >= 0)]
                # formatting the output
                df_agg.drop('Altitude', axis=1)
                df_agg['op'] = op
                df_agg['Thing'] = thing
                df_agg = df_agg[['op', 'Thing', 'Longitude', 'Latitude', 'Result']]

                if os.path.exists(output_path + csv_file):
                    df_agg.to_csv(output_path + csv_file, sep=';', header=False, mode='a', index=False)
                else:
                    df_agg.to_csv(output_path + csv_file, sep=';', header=True, mode='a', index=False)
    print(f'\t\t\tProcess done on {csv_file}, {combined_count} records aggregated')
