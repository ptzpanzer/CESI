import os
import gzip
import shutil
import multiprocessing
import xarray as xr
import pandas as pd
import math

# path contains downloaded .gz files
gz_folder = '../ABO_download/downloaded_files/'
output_folder = "./ABO/Dataset_Separation/"


def process_gz_file(gz_file):
    try:
        with gzip.open(gz_folder + gz_file, 'rb') as f:
            if f is not None:
                ds = xr.open_dataset(f, engine='scipy')
                process_ds(ds, gz_file)
    except Exception as e:
        print(f"\t\tError on file {gz_file}: {e}")


def process_ds(ds, gz_file):
    columns = ["latitude", "longitude", "altitude", "temperature", "windDir", "windSpeed"]

    data = {}
    for col in columns:
        data[col] = ds[col]

    df = pd.DataFrame(data)

    # cut studied area
    min_lon = -77
    max_lon = -74
    min_lat = 39
    max_lat = 42
    res_lon = 250
    res_lat = 250

    df_filtered = df.dropna(subset=["temperature", "windDir", "windSpeed"], how='all')
    df_filtered = df_filtered[(df_filtered['longitude'] >= min_lon) & (df_filtered['longitude'] <= max_lon) & (
                df_filtered['latitude'] >= min_lat) & (df_filtered['latitude'] <= max_lat)]
    df_filtered = df_filtered.reset_index(drop=True)

    # preprocessing
    #    - turn Lon/Lat to x/y pixels
    df_filtered['Longitude'] = (((df_filtered['longitude'] - min_lon) / (max_lon - min_lon)) * res_lon).apply(
        math.floor)
    df_filtered['Latitude'] = (((df_filtered['latitude'] - min_lat) / (max_lat - min_lat)) * res_lat).apply(math.floor)
    df_filtered = df_filtered[['Longitude', 'Latitude', 'altitude', "temperature", "windDir", "windSpeed"]]

    # data aggregation
    df_agg = df_filtered.groupby(['Longitude', 'Latitude', 'altitude']).agg('mean').reset_index()

    # to narrow format
    df_melted = pd.melt(df_agg, id_vars=['Longitude', 'Latitude', 'altitude'], var_name='op', value_name='Result')
    df_melted = df_melted.dropna(subset=["Result"])
    df_melted["Thing"] = 1

    df_melted = df_melted[['op', "Thing", 'Longitude', 'Latitude', 'altitude', "Result"]]

    dic_op_range = {
        "temperature": [204, 306],
        "windDir": [0, 360],
        "windSpeed": [0, 90],
    }

    cleaned_df = []
    for op in dic_op_range.keys():
        df_op = df_melted[df_melted['op'] == op]
        df_op = df_op[(df_op['Result'] >= dic_op_range[op][0]) & (df_op['Result'] < dic_op_range[op][1])]
        cleaned_df.append(df_op)

    df_out = pd.concat(cleaned_df, axis=0)

    df_out.to_csv(output_folder + gz_file[:-2] + "csv", index=False, sep=';')
    print(f"\tFile {gz_file} Done!")


def main():
    gz_files = [file for file in os.listdir(gz_folder) if file.endswith('.gz')]

    pool = multiprocessing.Pool(processes=2)
    pool.map(process_gz_file, gz_files)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
