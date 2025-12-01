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


def set_grid(d, res_x, res_y):
    rtn = [[0 for y in range(res_y)] for x in range(res_x)]
    for index, row in d.iterrows():
        if 0 <= row['Longitude'] < res_x and 0 <= row['Latitude'] <= res_y:
            x = int(row['Longitude'])
            y = int(row['Latitude'])
            rtn[x][y] += 1
    return np.matrix(rtn)


def matrix_analyze(m, res_x, res_y):
    _x = []
    _y = []
    width = 1
    depth = 1
    top = []
    for i in range(res_x):
        for j in range(res_y):
            if m.item((i, j)) != 0:
                _x.append(i)
                _y.append(j)
                top.append(m.item((i, j)))
    return np.array(_x), np.array(_y), np.zeros_like(np.array(top)), width, depth, np.array(top)


def visualize(in_path, out_path):
    geo_file = gdal.Open('./Data_Folder/CWSL_resampled.tif')
    res_x = geo_file.RasterXSize
    res_y = geo_file.RasterYSize

    min_x = res_x
    min_y = res_y
    max_x = 0
    max_y = 0

    folders = []
    folders_port = os.listdir(in_path)
    for folder in folders_port:
        folders.append([in_path, folder])

    dict_op_grid = {}
    for path, folder in folders:
        files = os.listdir(path + folder + '/')
        for file in files:
            print(f'Playing with file: {folder} + {file}')
            df = pd.read_csv(path + folder + '/' + file, sep=';')
            grid = set_grid(df, res_x, res_y)
            if file not in dict_op_grid.keys():
                tmp = [[0 for y in range(res_y)] for x in range(res_x)]
                dict_op_grid[file] = np.matrix(tmp)
            dict_op_grid[file] += grid

            if file != 'p_label.csv':
                df_max_x = df['Longitude'].max()
                df_min_x = df['Longitude'].min()
                df_max_y = df['Latitude'].max()
                df_min_y = df['Latitude'].min()

                if df_max_x > max_x:
                    max_x = df_max_x
                if df_max_y > max_y:
                    max_y = df_max_y
                if df_min_x < min_x:
                    min_x = df_min_x
                if df_min_y < min_y:
                    min_y = df_min_y

            # break

    total = np.matrix([[0 for y in range(res_y)] for x in range(res_x)])
    fig = plt.figure(1, figsize=(16, 16))
    for ob in dict_op_grid.keys():
        total += dict_op_grid[ob]
        x, y, z, w, d, t = matrix_analyze(dict_op_grid[ob], res_x, res_y)
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
        ax.bar3d(x, y, z, w, d, t, shade=True)
        plt.savefig(out_path + f'{ob}_{res_x}_{res_y}.png')
    x, y, z, w, d, t = matrix_analyze(total, res_x, res_y)
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
    ax.bar3d(x, y, z, w, d, t, shade=True)
    plt.savefig(out_path + f'All_{res_x}_{res_y}.png')

    print(f'min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}')
    with open(out_path + 'boundary.info', 'wb') as f:
        pickle.dump([min_x, max_x, min_y, max_y], f)
