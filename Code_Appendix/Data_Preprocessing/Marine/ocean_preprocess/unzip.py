import os
import zipfile
import shutil
import util
import random
import pickle
import pandas as pd
from datetime import datetime


def unzip_and_remove(zip_dir, target_dir):
    util.build_folder_and_clean(target_dir)

    zip_files = []
    for file in os.listdir(zip_dir):
        if ".zip" in file:
            zip_files.append(file)
    zip_files.sort()

    for file in zip_files:
        print(f"Unzipping file: {file}")
        with zipfile.ZipFile(zip_dir + file, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

    for file in os.listdir(target_dir):
        if 'data-policy' in file:
            os.remove(target_dir + file)


def csv_scan(target_dir, log_dir):
    util.build_folder_and_clean(log_dir)

    observed_variable_units_combinations = set()
    platform_station_combinations = set()
    total_count = 0
    failed_quality_flag_count = 0
    for file in os.listdir(target_dir):
        if 'csv-obs' in file:
            print(f"Scanning file: {file}")
            df = pd.read_csv(target_dir + file)
            observed_variable_units_combinations.update(
                zip(df['observed_variable'], df['units'])
            )
            total_count += df.shape[0]
            failed_quality_flag_count += df[df['quality_flag'] == 'Failed'].shape[0]
            platform_station_combinations.update(
                zip(df['platform_type'], df['station_type'])
            )

    output_file_path = log_dir + 'statistics_results.txt'
    with open(output_file_path, 'w') as f:
        f.write("1. Unique combinations of observed_variable and units:\n")
        for element in observed_variable_units_combinations:
            f.write(f"\t{element}\n")
        f.write("2. Unique combinations of platform_type and station_type:\n")
        for element in platform_station_combinations:
            f.write(f"\t{element}\n")
        f.write(f"3. Total number of rows: {total_count}\n")
        f.write(f"4. Total number of rows with quality_flag set to 'Failed': {failed_quality_flag_count}\n")


def csv_convert(target_dir, final_dir, a):
    util.build_folder_and_clean(final_dir)

    num_regions = 6
    region_sets = {i: {'train': set(), 'eval': set(), 'test': set(), "files": []} for i in range(num_regions)}
    for i in range(num_regions):
        region_sets[i]['eval'].add(i)

        test_regions = random.sample([x for x in range(num_regions) if x != i], 2)
        region_sets[i]['test'].update(test_regions)

        train_regions = set(range(num_regions)) - region_sets[i]['eval'] - region_sets[i]['test']
        region_sets[i]['train'].update(train_regions)
    with open("./OceanAt_res250/log.pkl", 'wb') as file:
        pickle.dump(region_sets, file)

    day_part = 6
    part_hours = 24 // day_part
    for file in os.listdir(target_dir):
        if 'csv-obs' in file:
            file_date = file.split("_")[2]
            df = pd.read_csv(target_dir + file)
            for h in range(0, day_part):
                print(f"Converting file: {file}-{h}")
                start_time_str = f'{file_date} {h*part_hours:02d}:00:00+00:00'
                end_time_str = f'{file_date} {(h+1)*part_hours:02d}:00:00+00:00'
                filtered_df = df[(df['latitude'] >= a[0]) & (df['latitude'] < a[1]) &
                                 (df['longitude'] >= a[2]) & (df['longitude'] < a[3]) &
                                 (df['date_time'] >= start_time_str) & (df['date_time'] < end_time_str) &
                                 (df['quality_flag'] != 'Failed')].copy()
                if len(filtered_df) == 0:
                    continue

                date_str = filtered_df['date_time'].apply(
                    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S+00:00").strftime("%Y-%m-%d")
                )
                if len(set(date_str)) > 1:
                    raise ValueError("Not all rows are from the same day.")

                # Output CSV contains following columns:(op, Thing, Longitude, Latitude, Result)
                filtered_df['op'] = filtered_df['observed_variable']
                thing_mapping = {
                    ('Ship', 'Sea station'): 1,
                    ('Lightship', 'Sea station'): 2,
                    ('Drifting buoy (of drifter)', 'Sea station'): 3
                }
                filtered_df['Thing'] = filtered_df.apply(
                    lambda row: thing_mapping.get((row['platform_type'], row['station_type'])), axis=1
                )
                filtered_df['Latitude'] = ((filtered_df['latitude'] - a[0]) / 0.1).astype(int)
                filtered_df['Longitude'] = ((filtered_df['longitude'] - a[2]) / 0.1).astype(int)
                filtered_df['Result'] = filtered_df['observation_value']
                result_df = filtered_df.groupby(
                    ['op', 'Thing', 'Longitude', 'Latitude', ]
                ).agg({'Result': 'mean'}).reset_index()

                water_temp_rows = result_df[result_df['op'] == 'water_temperature']
                for index, row in water_temp_rows.iterrows():
                    region = get_region(row['Longitude'], row['Latitude'])
                    new_op = f'water_temperature_{region}'
                    result_df.at[index, 'op'] = new_op

                dic_region_count = {}
                for region in range(num_regions):
                    dic_region_count[region] = result_df[result_df['op'] == f'water_temperature_{region}'].shape[0]
                dic_region_count["aux"] = result_df[~result_df['op'].str.contains('water_temperature_')].shape[0]

                keep = True
                for holdout in range(num_regions):
                    train_count = 0
                    for train in region_sets[holdout]["train"]:
                        train_count += dic_region_count[train]

                    test_count = 0
                    for test in region_sets[holdout]["test"]:
                        test_count += dic_region_count[test]

                    eval_count = 0
                    for evaluation in region_sets[holdout]["eval"]:
                        eval_count += dic_region_count[evaluation]

                    aux_count = dic_region_count["aux"]

                    if train_count <= 5 or test_count <= 2 or eval_count <= 2 or aux_count <= 10:
                        keep = False

                if keep:
                    result_df.to_csv(final_dir + f"{file_date}_{h:02d}.csv", header=True, index=False, sep=';')


def get_region(longitude, latitude):
    x_region = int(longitude / 250 * 3)
    y_region = int(latitude / 250 * 2)
    if x_region < 0 or x_region > 2 or y_region < 0 or y_region > 1:
        return None
    region = y_region * 3 + x_region
    return region


if __name__ == "__main__":
    zip_directory = "./Data/"
    target_directory = "./Ocean_ori/"
    final_directory = "./OceanAt_res250/Dataset_Separation/"
    log_directory = "./Ocean_log/"
    divide_directory = "./OceanAt_res250/Dataset_Separation/"
    area = [31, 56, -36, -11]

    # unzip downloaded file and remove unused files
    unzip_and_remove(zip_directory, target_directory)
    # analyze dataset metadata
    csv_scan(target_directory, log_directory)
    # cut, aggregate, and Format Transform
    csv_convert(target_directory, final_directory, area)

