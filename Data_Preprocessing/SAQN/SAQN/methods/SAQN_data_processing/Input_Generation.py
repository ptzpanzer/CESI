import os
import random
import pickle
import pandas as pd
from osgeo import gdal, osr
import numpy as np

import methods.data_processing.Manager_Folder as Manager_Folder


def Geotiff_norm(in_path, out_path):
    input_im = gdal.Open(in_path)
    data = []
    for i in range(input_im.RasterCount):
        input_im_band = input_im.GetRasterBand(i+1)
        stats = input_im_band.GetStatistics(False, True)
        min_value, max_value = stats[0], stats[1]
        input_im_band_ar = input_im.GetRasterBand(i+1).ReadAsArray()
        output_im_band_ar = (input_im_band_ar - min_value) / (max_value - min_value)
        data.append(output_im_band_ar.copy())

    output_file = out_path
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_file,
                           input_im.RasterXSize,
                           input_im.RasterYSize,
                           input_im.RasterCount,
                           gdal.GDT_Float32)
    for i in range(input_im.RasterCount):
        dst_ds.GetRasterBand(i+1).WriteArray(data[i])

    dst_ds.SetGeoTransform(input_im.GetGeoTransform())
    wkt = input_im.GetProjection()
    # setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())

    input_im = None
    dst_ds = None


def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])
    return np.hypot(d0, d1)


def interpolate(df, res_x, res_y):
    x = df['Longitude'].to_numpy()
    y = df['Latitude'].to_numpy()
    z = df['Result'].to_numpy()
    xi = np.linspace(0, res_x - 1, res_x)
    yi = np.linspace(0, res_y - 1, res_y)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()

    dist = distance_matrix(x, y, xi, yi)

    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x, y, x, y)

    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)

    # Multiply the weights for each interpolated point by the distances
    zi = np.dot(dist.T, weights)
    return zi.reshape((res_x, res_y))


def SAQN_norm(in_path, mid_path, out_path):
    geo_file = gdal.Open('./Data_Folder/Dataset_Mid/CWSL_norm.tif')
    res_x = geo_file.RasterXSize
    res_y = geo_file.RasterYSize

    folders = os.listdir(in_path)
    random.shuffle(folders)

    train_folders = folders[:int(0.7 * len(folders))]
    test_folders = folders[int(0.7 * len(folders)):int(0.9 * len(folders))]
    eval_folders = folders[int(0.9 * len(folders)):]

    dic_op_minmax = {}
    for folder in train_folders:
        files = os.listdir(in_path + folder + '/')
        for file in files:
            print(f'Analyze - Playing with file: {folder} + {file}')
            df = pd.read_csv(in_path + folder + '/' + file, sep=';')
            if file in ['mcpm10.csv', 'p_label.csv', 's_label.csv']:
                op = 'mcpm10'
            else:
                op = file[:-4]
            if op not in dic_op_minmax.keys():
                dic_op_minmax[op] = [float("inf"), -1 * float("inf")]
            df_min = df['Result'].min()
            df_max = df['Result'].max()
            if dic_op_minmax[op][0] > df_min:
                dic_op_minmax[op][0] = df_min
            if dic_op_minmax[op][1] < df_max:
                dic_op_minmax[op][1] = df_max

    for folder in folders:
        files = os.listdir(in_path + folder + '/')
        inter_path = mid_path + folder + '/'
        Manager_Folder.build_folder_and_clean(inter_path)
        for file in files:
            print(f'Normalize - Playing with file: {folder} + {file}')
            df = pd.read_csv(in_path + folder + '/' + file, sep=';')
            if file in ['mcpm10.csv', 'p_label.csv', 's_label.csv']:
                op = 'mcpm10'
            else:
                op = file[:-4]
            df['Result'] = (df['Result'] - dic_op_minmax[op][0]) / (dic_op_minmax[op][1] - dic_op_minmax[op][0])
            df.to_csv(inter_path + file, sep=',', header=True, mode='a', index=False)

    dic_train_size = {}
    dic_test_size = {}
    dic_eval_size = {}
    for folder in folders:
        flag = True
        files = os.listdir(mid_path + folder + '/')
        dic_op_map = {}
        input_df_list = []
        label_df_list = []
        for file in files:
            print(f'Generate - Playing with file: {folder} + {file}')
            df = pd.read_csv(mid_path + folder + '/' + file, sep=',')
            if file not in ['mcpm10.csv', 'p_label.csv', 's_label.csv'] and len(df) > 1:
                dic_op_map[file[:-4]] = interpolate(df, res_x, res_y)
            elif file not in ['mcpm10.csv', 'p_label.csv', 's_label.csv'] and len(df) <= 1:
                flag = False
                break
            else:
                if folder in train_folders:
                    if file == 'mcpm10.csv':
                        df.insert(2, 'Port', 0.0)
                        input_df_list.append(df)
                    elif file == 's_label.csv':
                        df.insert(2, 'Port', 0.0)
                        label_df_list.append(df)
                    elif file == 'p_label.csv':
                        df.insert(2, 'Port', 1.0)
                        # input_part_50 = df.sample(frac=0.5)
                        # input_df_list.append(input_part_50)
                        # label_part_50 = df.drop(input_part_50.index)
                        # label_df_list.append(label_part_50)
                        input_df_list.append(df)
                else:
                    if file == 'mcpm10.csv':
                        df.insert(2, 'Port', 0.0)
                        input_df_list.append(df)
                    elif file == 's_label.csv':
                        df.insert(2, 'Port', 0.0)
                        label_df_list.append(df)
                    elif file == 'p_label.csv':
                        df.insert(2, 'Port', 1.0)
                        # label_df_list.append(df)
                        input_df_list.append(df)

        if not flag:
            continue
        else:
            df_input = pd.concat(input_df_list)
            df_label = pd.concat(label_df_list)

            scene_size = len(df_input) + len(df_label)
            if folder in train_folders:
                dic_train_size[folder] = scene_size
            elif folder in test_folders:
                dic_test_size[folder] = scene_size
            elif folder in eval_folders:
                dic_eval_size[folder] = scene_size

            output_file = out_path + folder + '.tif'
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(output_file,
                                   res_x,
                                   res_y,
                                   geo_file.RasterCount + len(dic_op_map.keys()),
                                   gdal.GDT_Float32)
            for i in range(geo_file.RasterCount):
                geo_file_band_ar = geo_file.GetRasterBand(i + 1).ReadAsArray()
                dst_ds.GetRasterBand(i + 1).WriteArray(geo_file_band_ar)
            list_keys = list(dic_op_map.keys())
            list_keys.sort()
            for i in range(len(list_keys)):
                dst_ds.GetRasterBand(i + 1 + geo_file.RasterCount).WriteArray(dic_op_map[list_keys[i]])

            dst_ds.SetGeoTransform(geo_file.GetGeoTransform())
            wkt = geo_file.GetProjection()
            # setting spatial reference of output raster
            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)
            dst_ds.SetProjection(srs.ExportToWkt())

            dst_ds = None

            df_input.to_csv(out_path + folder + '_input.csv', sep=';', header=True, mode='w', index=False)
            df_label.to_csv(out_path + folder + '_label.csv', sep=';', header=True, mode='w', index=False)

    dic_train_size = {k: v for k, v in sorted(dic_train_size.items(), key=lambda item: item[1], reverse=True)}
    dic_test_size = {k: v for k, v in sorted(dic_test_size.items(), key=lambda item: item[1], reverse=True)}
    dic_eval_size = {k: v for k, v in sorted(dic_eval_size.items(), key=lambda item: item[1], reverse=True)}
    train_list = list(dic_train_size.keys())
    test_list = list(dic_test_size.keys())
    eval_list = list(dic_eval_size.keys())

    with open(out_path + 'norm.info', 'wb') as f:
        pickle.dump(dic_op_minmax, f)
    with open(out_path + 'divide_set.info', 'wb') as f:
        pickle.dump([train_list, test_list, eval_list], f)

    geo_file = None


def check_and_append(path, df):
    existed = os.path.exists(path)
    if existed:
        df.to_csv(path, sep=';', header=False, mode='a', index=False)
    else:
        df.to_csv(path, sep=';', header=True, mode='w', index=False)


def SAQN_norm_approx(in_path, out_path):
    # do the following for each fold
    for i in range(4):
        # Random one station from the rest 3 stations in this fold for testing
        my_list = [0, 1, 2, 3]
        my_list.remove(i)
        test_station = random.choice(my_list)
        print(f'Station {test_station} is chosen as tester in fold {i}')

        # separate time slices into 7:2:1 randomly
        folders = list(os.listdir(in_path))
        folders.sort()
        folders_train = folders[:int(0.7 * len(folders))]
        folders_test = folders[int(0.7 * len(folders)):int(0.9 * len(folders))]
        folders_eval = folders[int(0.9 * len(folders)):]
        # dump the result
        with open(out_path + f'divide_set_{i}.info', 'wb') as f:
            pickle.dump([folders_train, folders_test, folders_eval, test_station], f)

        # Analyze info for Z-Score in training set only [0.7*T and 2/4 Labels]
        dic_op_df = {}
        for folder in folders_train:
            files = os.listdir(in_path + folder + '/')
            for file in files:
                # skip station i if it is i-th fold
                if file == f's_label_{i}.csv':
                    print(f'Skip: {in_path} + {folder} + {file} since is {i}-th fold')
                    continue
                if file == f's_label_{test_station}.csv':
                    print(f'Skip: {in_path} + {folder} + {file} since is selected as test station')
                    continue
                # analyze the rest files
                print(f'Analyze - Playing with file: {in_path} + {folder} + {file}')
                df = pd.read_csv(in_path + folder + '/' + file, sep=';')
                if file in ['mcpm10.csv', 'p_label.csv', 's_label_0.csv',
                            's_label_1.csv', 's_label_2.csv', 's_label_3.csv']:
                    op = 'mcpm10'
                else:
                    op = file[:-4]
                if op not in dic_op_df.keys():
                    dic_op_df[op] = df
                else:
                    dic_op_df[op] = pd.concat([dic_op_df[op], df])
        # calculate mean and std then dump result
        dic_op_meanstd = {}
        with open(out_path + f'norm_{i}.log', 'w') as f:
            for op in dic_op_df.keys():
                op_mean = dic_op_df[op]['Result'].mean()
                op_std = dic_op_df[op]['Result'].std()
                dic_op_meanstd[op] = [op_mean, op_std]
                f.write(f'norm log for op: {op} ---- len: {len(dic_op_df[op])}, mean: {op_mean}, std: {op_std}\n')
        with open(out_path + f'norm_{i}.info', 'wb') as f:
            pickle.dump(dic_op_meanstd, f)

        # do the Z-Score
        out_path_i = out_path + f'{i}/'
        Manager_Folder.build_folder_and_clean(out_path_i)
        folders = list(os.listdir(in_path))
        for folder in folders:
            files = os.listdir(in_path + folder + '/')
            scene_df_list = []
            for file in files:
                print(f'Normalize - Playing with file: {in_path} + {folder} + {file}')
                df = pd.read_csv(in_path + folder + '/' + file, sep=';')
                if file in \
                        ['mcpm10.csv', 'p_label.csv', 's_label_0.csv',
                         's_label_1.csv', 's_label_2.csv', 's_label_3.csv']:
                    op = 'mcpm10'
                else:
                    op = file[:-4]
                df['Result'] = (df['Result'] - dic_op_meanstd[op][0]) / dic_op_meanstd[op][1]
                df.insert(0, 'op', file[:-4])
                scene_df_list.append(df)
            df_out = pd.concat(scene_df_list)
            df_out.to_csv(out_path_i + folder + '.csv', sep=';', header=True, mode='w', index=False)


def translate_scene(scene, in_path, out_path, work_style, fold, test_station):
    df = pd.read_csv(in_path + scene + '.csv', sep=';')
    if work_style == 'train':
        allow_label = ['s_label_0', 's_label_1', 's_label_2', 's_label_3']
        allow_label.remove(f's_label_{fold}')
        allow_label.remove(f's_label_{test_station}')
    elif work_style == 'test':
        allow_label = [f's_label_{test_station}', ]
    elif work_style == 'eval':
        allow_label = [f's_label_{fold}', ]
    df_label = df.loc[df['op'].isin(allow_label)]
    df_label['Scene'] = scene
    df_label['op'] = 's_label'
    check_and_append(out_path + work_style + f'.csv', df_label)
