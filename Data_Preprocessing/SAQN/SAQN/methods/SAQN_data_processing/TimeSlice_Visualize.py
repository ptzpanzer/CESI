# Draw data distributions on Grid map, help to decide AABB of the interested zone
#   - Separation:
#       - Files in ./Dataset_Aggregated/
#   - Output:
#       - Grid maps of data distributions in ./Dataset_Figures/
#       - Final AABB of the interested zone saved in code

import os
import pandas as pd
import numpy as np
from osgeo import gdal
import pickle
import matplotlib.pyplot as plt
import random


def set_grid(d):
    _x = []
    _y = []
    width = 1
    depth = 1
    top = []
    for index, row in d.iterrows():
        _x.append(int(row['Longitude']))
        _y.append(int(row['Latitude']))
        top.append(row['Result'])

    return np.array(_x), np.array(_y), np.zeros_like(np.array(top)), width, depth, np.array(top)


def visualize(in_path, out_path):
    geo_file = gdal.Open(in_path + 'CWSL_resampled.tif')
    res_x = geo_file.RasterXSize
    res_y = geo_file.RasterYSize

    files = os.listdir(in_path + 'Dataset_Separation/')

    for file in files:
        df = pd.read_csv(in_path + 'Dataset_Separation/' + file, sep=';')
        df = df.replace(['p_label', 's_label_0', 's_label_1', 's_label_2', 's_label_3'], 'mcpm10')

        for op in df['op'].unique():
            df_op = df[df['op'] == op]
            x, y, z, w, d, t = set_grid(df_op)

            fig = plt.figure(1, figsize=(16, 16))
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(0, res_x)
            ax.set_ylim3d(0, res_y)
            major_ticks = np.arange(0, res_x, 100)
            minor_ticks = np.arange(0, res_x, 10)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            major_ticks = np.arange(0, res_y, 100)
            minor_ticks = np.arange(0, res_y, 10)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)
            ax.bar3d(x, y, z, w, d, t, shade=True, alpha=0.3)
            plt.savefig(out_path + f'{file[:-4]}_{op}.png')
