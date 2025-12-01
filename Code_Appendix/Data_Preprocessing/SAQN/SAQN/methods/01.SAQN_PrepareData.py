import os
import pickle
import shutil

from multiprocessing import Process
from multiprocessing import Pool
from osgeo import gdal
from dateutil.relativedelta import relativedelta
import dateutil.parser

import methods.SAQN_data_processing.Manager_Settings as Manager_Settings
import methods.SAQN_data_processing.Manager_Folder as Manager_Folder

import methods.SAQN_data_processing.METO_Raw_Standarizer as METO_Raw_Standarizer
import methods.SAQN_data_processing.Boundary_Analyzer_0 as Boundary_Analyzer_0
import methods.SAQN_data_processing.Boundary_Analyzer_1 as Boundary_Analyzer_1
import methods.SAQN_data_processing.Data_Filter as Data_Filter
import methods.SAQN_data_processing.Landuse_Filter as Landuse_Filter
import methods.SAQN_data_processing.Data_Aggregator as Data_Aggregator
import methods.SAQN_data_processing.DataSlice_Visualize as DataSlice_Visualize
import methods.SAQN_data_processing.Dataset_Separation as Dataset_Separation
import methods.SAQN_data_processing.Input_Generation as Input_Generation


def Data_Analyse(SAQN_csv_origin_path, METO_csv_origin_path):
    # Read SAQN Boundary Analyzer Settings
    SAQN_analyzer_setting_path = 'data_processing/Boundary_Analyzer_Settings_SAQN.txt'
    SAQN_analyzer_settings = Manager_Settings.read_boundary_analyzer_setting(SAQN_analyzer_setting_path)
    # Run SAQN Boundary Analyzer Phase 0
    SAQN_csv_interested_path = './Data_Folder/SAQN2021_CSV_Interested/'
    Manager_Folder.build_folder_and_clean(SAQN_csv_interested_path)
    SAQN_analysis_result_path = './Data_Folder/SAQN2021_Analysis_DataValue/'
    Manager_Folder.build_folder_and_clean(SAQN_analysis_result_path)
    Boundary_Analyzer_0.boundary_analyzer_0(
        SAQN_csv_origin_path, SAQN_analyzer_settings, SAQN_csv_interested_path, SAQN_analysis_result_path
    )

    # Run METO Boundary Analyzer Phase 0
    METO_csv_decompressed_path = './Data_Folder/METO2021_CSV_Decompressed/'
    Manager_Folder.build_folder_and_clean(METO_csv_decompressed_path)
    METO_Raw_Standarizer.raw_standarizer(METO_csv_origin_path, METO_csv_decompressed_path, 2017, 2022)
    # Read METO Boundary Analyzer Settings
    METO_analyzer_setting_path = 'data_processing/Boundary_Analyzer_Settings_METO.txt'
    METO_analyzer_settings = Manager_Settings.read_boundary_analyzer_setting(METO_analyzer_setting_path)
    # Run METO Boundary Analyzer Phase 0
    METO_csv_interested_path = './Data_Folder/METO2021_CSV_Interested/'
    Manager_Folder.build_folder_and_clean(METO_csv_interested_path)
    METO_analysis_result_path = './Data_Folder/METO2021_Analysis_DataValue/'
    Manager_Folder.build_folder_and_clean(METO_analysis_result_path)
    Boundary_Analyzer_0.boundary_analyzer_0(
        METO_csv_decompressed_path, METO_analyzer_settings, METO_csv_interested_path, METO_analysis_result_path
    )


def Boundary_choice():
    SAQN_analysis_result_path = './Data_Folder/SAQN2021_Analysis_DataValue/'
    METO_analysis_result_path = './Data_Folder/METO2021_Analysis_DataValue/'
    # Run Boundary Analyzer Phase 1
    Manager_Folder.build_folder_and_clean(Analysis_result_combined_path)
    Boundary_Analyzer_1.boundary_analyzer_1(
        [SAQN_analysis_result_path, METO_analysis_result_path], Analysis_result_combined_path
    )


def Data_abnormal_drop(csv_path, csv_file):
    bound_setting_path = Analysis_result_combined_path + 'Data_bounds.txt'
    bound_settings = Manager_Settings.read_filter_bound_setting(bound_setting_path)
    # Run Data Filter
    return Data_Filter.data_filter(csv_path, csv_file, bound_settings)


def Folder_cleanup():
    SAQN_analysis_result_path = './Data_Folder/SAQN2021_Analysis_DataValue/'
    METO_analysis_result_path = './Data_Folder/METO2021_Analysis_DataValue/'
    METO_csv_decompressed_path = './Data_Folder/METO2021_CSV_Decompressed/'
    # Do the Clean-up
    Manager_Folder.folder_destroy(SAQN_analysis_result_path)
    Manager_Folder.folder_destroy(METO_analysis_result_path)
    Manager_Folder.folder_destroy(SAQN_csv_interested_path)
    Manager_Folder.folder_destroy(METO_csv_decompressed_path)
    Manager_Folder.folder_destroy(METO_csv_interested_path)
    Manager_Folder.folder_destroy(Analysis_result_combined_path)


def CWSL_data_preprocess(CWSL_tif_origin_path):
    used_files = ['IMD_2018_010m.tif', 'TCD_2018_010m.tif', 'WWPI_2018_010m.tif']
    out_path = './Data_Folder/CWSL2018_TIF_Valid/'
    Manager_Folder.build_folder_and_clean(out_path)
    Landuse_Filter.cut_map(CWSL_tif_origin_path, used_files, out_path + 'CWSL.tif')


def separation():
    separation_in_path = './Data_Folder/Dataset_Aggregated/'
    separation_out_path = './Data_Folder/Dataset_Separation/'
    Manager_Folder.build_folder_and_clean(separation_out_path)
    Dataset_Separation.dataset_separation_new(separation_in_path, separation_out_path)


def visualize():
    visualize_in_path = './Data_Folder/Dataset_Separation/'
    visualize_out_path = './Data_Folder/Dataset_Visualize/'
    Manager_Folder.build_folder_and_clean(visualize_out_path)
    DataSlice_Visualize.visualize(visualize_in_path, visualize_out_path)


# def generation():
#     generation_in_path = './Data_Folder/Dataset_Separation/Eval/'
#     generation_out_path = './Data_Folder/Dataset_Mid/'
#     Manager_Folder.build_folder_and_clean(generation_out_path)
#     Input_Generation.Geotiff_norm('./Data_Folder/CWSL_resampled.tif', generation_out_path + 'CWSL_norm.tif')
#
#     final_out_path = './Data_Folder/Input/'
#     Manager_Folder.build_folder_and_clean(final_out_path)
#     Input_Generation.SAQN_norm(generation_in_path, generation_out_path, final_out_path)
#     Manager_Folder.folder_destroy(generation_out_path)
#
#
# def approx_generation():
#     generation_in_path = './Data_Folder/Dataset_Separation/'
#     final_out_path = './Data_Folder/Input_Approx_250uv/'
#     Manager_Folder.build_folder_and_clean(final_out_path)
#
#     Input_Generation.Geotiff_norm('./Data_Folder/CWSL_resampled.tif', final_out_path + 'CWSL_norm.tif')
#     Input_Generation.SAQN_norm_approx(generation_in_path, final_out_path)


def approx_generation_new():
    generation_in_path = './Data_Folder/Dataset_Separation/'
    final_out_path = './Data_Folder/Input_Approx_500uv/'
    Manager_Folder.build_folder_and_clean(final_out_path)

    shutil.copyfile('./Data_Folder/CWSL_resampled.tif', final_out_path + 'CWSL_resampled.tif')
    Input_Generation.SAQN_norm_approx(generation_in_path, final_out_path)


def translate_dataset():
    in_path_all = './Data_Folder/Input_Approx_250uv/'
    out_path_all = './Data_Folder/Trans_Call_res250uv/'
    Manager_Folder.build_folder_and_clean(out_path_all)
    for fold in range(4):
        in_path = in_path_all + f'{fold}/'
        out_path = out_path_all + f'{fold}/'
        Manager_Folder.build_folder_and_clean(out_path)
        with open(in_path_all + f'divide_set_{fold}.info', 'rb') as f:
            divide_set = pickle.load(f)
        train_scenes = divide_set[0]
        test_scenes = divide_set[1]
        eval_scenes = divide_set[2]
        test_station = divide_set[3]
        for i in range(len(train_scenes)):
            print(f'Working on train scenes {i + 1}/{len(train_scenes)}')
            Input_Generation.translate_scene(train_scenes[i], in_path, out_path, 'train', fold, test_station)
        for i in range(len(test_scenes)):
            print(f'Working on test scenes {i + 1}/{len(test_scenes)}')
            Input_Generation.translate_scene(test_scenes[i], in_path, out_path, 'test', fold, test_station)
        for i in range(len(eval_scenes)):
            print(f'Working on eval scenes {i + 1}/{len(eval_scenes)}')
            Input_Generation.translate_scene(eval_scenes[i], in_path, out_path, 'eval', fold, test_station)


if __name__ == '__main__':
    # Analyse SAQN and METO Dataset
    Data_Analyse(
        '../../0.SmartAQnet_Dataset/SAQN202112_CSV_Decompressed/',
        '../../0.SmartAQnet_Dataset/Meteo202205_JSON_Decompressed/data/'
    )
    # Select boundary that defines abnormal values
    Boundary_choice()

    # # Multi-processing to drop abnormal values
    SAQN_csv_interested_path = './Data_Folder/SAQN2021_CSV_Interested/'
    METO_csv_interested_path = './Data_Folder/METO2021_CSV_Interested/'
    Analysis_result_combined_path = './Data_Folder/Analysis_DataValue_Combined/'
    TOTAL_csv_valid_path = './Data_Folder/Total_CSV_Valid/'
    Manager_Folder.build_folder_and_clean(TOTAL_csv_valid_path)
    files_list = []
    for path in [SAQN_csv_interested_path, METO_csv_interested_path]:
        files_in_path = os.listdir(path)
        for file in files_in_path:
            files_list.append((path, file))
    for path, file in files_list:
        print(f'Dropping abnormal value on {path}{file}.')
        data_out = Data_abnormal_drop(path, file)
        process_list = []
        for time in data_out['Time'].unique():
            process_list.append((data_out, TOTAL_csv_valid_path, time))
        p = Pool(20)
        p.map(Data_Filter.multi_print, process_list)
        p.close()
        p.join()
        print(f'\t\tAll Writing for {path}{file} finished, proceeding to next file.')

    # Clean-up the mid-folders
    Folder_cleanup()

    # # prepare Land use dataset
    # CWSL_data_preprocess('../../0.SmartAQnet_Dataset/CWS2018/')
    # # Reset CWSL area and resolution
    # cut_x_min = 200
    # cut_y_min = 200
    # cut_x_range = 1000
    # cut_y_range = 1000
    # tgt_X_res = 250
    # tgt_Y_res = 250
    # # cut and save
    # geo_file = gdal.Open('./Data_Folder/CWSL2018_TIF_Valid/CWSL.tif')
    # gdal.Translate('./Data_Folder/CWSL_cut.tif', geo_file,
    #                srcWin=[cut_x_min, cut_y_min, cut_x_range, cut_y_range],
    #                options=['-a_scale', '1'])
    # # resample and save
    # geo_file = gdal.Open('./Data_Folder/CWSL_cut.tif')
    # gdal.Warp('./Data_Folder/CWSL_resampled.tif', geo_file, width=tgt_X_res, height=tgt_Y_res)

    # Multi-Process Temporal & Spatial aggregation
    TA_input_path = './Data_Folder/Total_CSV_Valid/'
    TA_output_path = './Data_Folder/Dataset_Aggregated/'
    Manager_Folder.build_folder_and_clean(TA_output_path)
    files = os.listdir(TA_input_path)
    process_list = []
    for file in files:
        process_list.append((TA_input_path, file, TA_output_path))
    p = Pool(20)
    p.map(Data_Aggregator.child_aggregator, process_list)
    p.close()
    p.join()
    print('Temporal & Spatial Aggregation All Finished')

    separation()
    # visualize()
    # approx_generation()
    # translate_dataset()
