import os
import pandas as pd

import methods.data_processing.Manager_Folder as Manager_Folder


def dataset_separation_new(in_path, out_path):
    label_device_stable_list = [
        ['saqn:t:b666034', 'saqn:t:umweltbundesamt.de:station_augsburg_lfu:deby099'],
        ['saqn:t:12b713d', 'saqn:t:umweltbundesamt.de:station_augsburg_bourges-platz:deby007'],
        ['saqn:t:4049564', 'saqn:t:umweltbundesamt.de:station_augsburg_karlstrasse:deby110'],
        ['saqn:t:d42cbb8', 'saqn:t:umweltbundesamt.de:station_augsburg_koenigsplatz:deby006']
    ]

    device_ranking = {
        0: [
            'saqn:t:82c620f', 'saqn:t:ce97848', 'saqn:t:d154ef6',
            'saqn:t:geo.uni-augsburg.de:Tasche:1', 'saqn:t:geo.uni-augsburg.de:Tasche:2',
            'saqn:t:geo.uni-augsburg.de:Tasche_2', 'saqn:t:geo.uni-augsburg.de:Tasche:3',
            'saqn:t:geo.uni-augsburg.de:Tasche:4', 'saqn:t:geo.uni-augsburg.de:Tasche:5',
            'saqn:t:geo.uni-augsburg.de:Tasche_5', 'saqn:t:geo.uni-augsburg.de:Tasche:6',
            'saqn:t:geo.uni-augsburg.de:Tasche:7', 'saqn:t:geo.uni-augsburg.de:Tasche:8',
            'saqn:t:geo.uni-augsburg.de:Tasche:9',
        ],
        1: [
            "saqn:t:7bd2cd3", "saqn:t:4d8e38a", "saqn:t:56cf860", "saqn:t:247632b", "saqn:t:ece3ca6",
            "saqn:t:ca57f32", "saqn:t:c3a8ab3", "saqn:t:77a3477", "saqn:t:bfc5b4d", "saqn:t:4e63052",
            "saqn:t:803206c", "saqn:t:57b8853", "saqn:t:f437b71", "saqn:t:teco.edu:crowdsensor:179880",
            "saqn:t:teco.edu:crowdsensor:14340632", "saqn:t:teco.edu:crowdsensor:179552", "saqn:t:9d5abc7",
            "saqn:t:teco.edu:crowdsensor:1163952", "saqn:t:teco.edu:crowdsensor:179992",
            "saqn:t:teco.edu:crowdsensor:16748656", "saqn:t:teco.edu:crowdsensor:16750008",
            "saqn:t:teco.edu:crowdsensor:16748228", "saqn:t:2cb209c", "saqn:t:f536ece", "saqn:t:a01385e",
            "saqn:t:744b88f", "saqn:t:3d63e0a", "saqn:t:43ae704", "saqn:t:f80f5d2", "saqn:t:e82832a",
            "saqn:t:812afed", "saqn:t:3dc97e0", "saqn:t:886f0bb", "saqn:t:2ea82d7", "saqn:t:9fe44e2",
            "saqn:t:3c004a9", "saqn:t:teco.edu:crowdsensor:4447764", "saqn:t:47b8b62", "saqn:t:c137644",
            "saqn:t:86115af", "saqn:t:efc7ebb", "saqn:t:ab82785", "saqn:t:c6f4ce8", "saqn:t:35da367",
            "saqn:t:fa8f4c5", "saqn:t:f54b814", "saqn:t:e69fe33", "saqn:t:6a6d2b9", "saqn:t:92ad07f",
            "saqn:t:1d70296", "saqn:t:62d4572", "saqn:t:6e4c022", "saqn:t:d4ada54", "saqn:t:dae547b",
            "saqn:t:0ff2988", "saqn:t:8e3b123", "saqn:t:178d962", "saqn:t:175482a", "saqn:t:3382f32",
            "saqn:t:7a3af55", "saqn:t:fc25866", "saqn:t:4a9f9bf", "saqn:t:8155445", "saqn:t:0b3da13",
            "saqn:t:47e9045", "saqn:t:e9ef626", "saqn:t:a3b182c", "saqn:t:4d28468", "saqn:t:052903d",
            "saqn:t:5e102be", "saqn:t:df1c3d7", "saqn:t:382ee70", "saqn:t:37b825b", "saqn:t:94d5923",
            "saqn:t:f48f90f", "saqn:t:1eec69c", "saqn:t:082196b", "saqn:t:c5f04b3", "saqn:t:00afbb4",
            "saqn:t:d6c4339", "saqn:t:154390b", "saqn:t:8b9b677", "saqn:t:b10758c", "saqn:t:cf02643",
            "saqn:t:114f0d7", "saqn:t:e9c313a", "saqn:t:ea4f9e0", "saqn:t:d718898", "saqn:t:54e61ee",
            "saqn:t:0e4afee", "saqn:t:87c1233", "saqn:t:4c18f4d", "saqn:t:teco.edu:crowdsensor:179188",
            "saqn:t:520e9bc", "saqn:t:teco.edu:crowdsensor:12886260",
        ],
        2: [
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17008", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17012",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17018", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17017",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17024", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17022",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17002", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17011",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17015", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17023",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17007", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17006",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17013", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17003",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17004", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17001",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17016", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17009",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17005", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17014",
            "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17010", "saqn:t:grimm-aerosol.com:EDM80NEPH:SN17019",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19001", "saqn:t:grimm-aerosol.com:edm80opc:sn19002",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19003", "saqn:t:grimm-aerosol.com:edm80opc:sn19004",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19005", "saqn:t:grimm-aerosol.com:edm80opc:sn19007",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19009", "saqn:t:grimm-aerosol.com:edm80opc:sn19010",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19011", "saqn:t:grimm-aerosol.com:edm80opc:sn19013",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19014", "saqn:t:grimm-aerosol.com:edm80opc:sn19015",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19016", "saqn:t:grimm-aerosol.com:edm80opc:sn19017",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19018", "saqn:t:grimm-aerosol.com:edm80opc:sn19021",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19019", "saqn:t:grimm-aerosol.com:edm80opc:sn19020",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19006", "saqn:t:grimm-aerosol.com:edm80opc:sn19050",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19051", "saqn:t:grimm-aerosol.com:edm80opc:sn19052",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19053", "saqn:t:grimm-aerosol.com:edm80opc:sn19054",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19022", "saqn:t:grimm-aerosol.com:edm80opc:sn19023",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19024", "saqn:t:grimm-aerosol.com:edm80opc:sn19025",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19026", "saqn:t:grimm-aerosol.com:edm80opc:sn19027",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19055", "saqn:t:grimm-aerosol.com:edm80opc:sn19056",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19057", "saqn:t:grimm-aerosol.com:edm80opc:sn19058",
            "saqn:t:grimm-aerosol.com:edm80opc:sn19059", "saqn:t:grimm-aerosol.com:edm80opc:sn19060",
            "saqn:t:helmholtz-muenchen.de:hmgu_epi_container:097467_1",
            "saqn:t:helmholtz-muenchen.de:hmgu_epi_container_meteo:097470_1",
            "kobelweg", "koenigsplatz", "rotes_tor", "stephan", "zentralklinikum", "uni", "uni2",
        ],
        3: label_device_stable_list[0] + label_device_stable_list[1] +
           label_device_stable_list[2] + label_device_stable_list[3] + [
               "saqn:t:grimm-aerosol.com:EDM164:7H170013", "saqn:t:grimm-aerosol.com:EDM164:7H180001",
               "saqn:t:grimm-aerosol.com:EDM164:7H180002", "saqn:t:grimm-aerosol.com:EDM164:7H170026",
               "saqn:t:grimm-aerosol.com:EDM164:@@@SRISS", "saqn:t:grimm-aerosol.com:EDM164:7H170014",
        ],
    }
    total_label_list = label_device_stable_list[0] + label_device_stable_list[1] + label_device_stable_list[2] + label_device_stable_list[3]
    op_list = ['mcpm10', 'mcpm2p5', 'ta', 'hur', 'plev', 'precip', 'wsx', 'wsy', 'globalrad']

    files = os.listdir(in_path)
    files.sort()
    for file in files:
        print(f'Playing with file: {file}')
        df = pd.read_csv(in_path + file, sep=';')

        # Analyze each time slice
        suitable_class = 'ok'
        #  - First, analyse the no-label devices
        #      - Filter readings out
        df_nolabel = df[~df['Thing'].isin(total_label_list)]
        #      - replace device name with their faithful rank
        for i in range(4):
            df_nolabel = df_nolabel.replace(device_ranking[i], i)
        #      - aggregate by rank and grid, decide the quality of the scenario
        nolabel_avg_list = []
        for op in op_list:
            df_nolabel_op = df_nolabel[df_nolabel['op'] == op]
            df_nolabel_op_avg = df_nolabel_op.groupby(['Thing', 'Longitude', 'Latitude'], as_index=False).mean(numeric_only=True)
            if op == 'mcpm10' and len(df_nolabel_op_avg) <= 5.0:
                suitable_class = 'no'
                print(f'\tRefused: Reference is too few. {op}')
                break
            if op != 'mcpm10' and len(df_nolabel_op_avg) <= 1.0:
                suitable_class = 'no'
                print(f'\tRefused: OP is not full. {op}')
                break
            nolabel_avg_list.append([op, df_nolabel_op_avg])

        #  - Then, analyse the stable label devices
        df_slabel_op_avg_list = []
        for i in range(4):
            df_slabel = df[df['Thing'].isin(label_device_stable_list[i])]
            df_slabel_op = df_slabel[df_slabel['op'] == 'mcpm10']
            for j in range(4):
                df_slabel_op = df_slabel_op.replace(device_ranking[j], j)
            df_slabel_op_avg = df_slabel_op.groupby(['Thing', 'Longitude', 'Latitude'], as_index=False).mean(numeric_only=True)
            if len(df_slabel_op_avg) < 1.0:
                suitable_class = 'no'
                print(f'\tRefused: Stable label is not full.')
                break
            df_slabel_op_avg_list.append(df_slabel_op_avg)

        # Sort according to suitable_class
        if suitable_class == 'no':
            continue
        elif suitable_class == 'ok':
            full_out_path = out_path + file
            header = 0
            for item in nolabel_avg_list:
                item[1].insert(0, 'op', item[0])
                if header == 0:
                    item[1].to_csv(full_out_path, sep=';', header=True, mode='a', index=False)
                    header = 1
                else:
                    item[1].to_csv(full_out_path, sep=';', header=False, mode='a', index=False)
            for i in range(4):
                df_slabel_op_avg_list[i].insert(0, 'op', f's_label_{i}')
                df_slabel_op_avg_list[i].to_csv(full_out_path, sep=';', header=False, mode='a', index=False)
