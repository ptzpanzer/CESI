import ast


# generate filter settings from setting file
def read_boundary_analyzer_setting(setting_file_path):
    dic_settings = {}

    with open(setting_file_path, 'r', encoding='utf-8') as f_setting:
        lines = f_setting.readlines()
        item = ''
        for line in lines:
            tmp = line.strip()
            if len(tmp) != 0:
                if '/*MEMO*/' in tmp:
                    continue
                else:
                    if '/*ITEM*/' in tmp:
                        item = tmp.replace('/*ITEM*/', '')
                        dic_settings[item] = []
                    else:
                        if item != 'operations':
                            dic_settings[item].append(tmp)
                        else:
                            if '/*DELT*/' in tmp:
                                th = tmp.replace('/*DELT*/', '').split('\t')[0]
                                op = tmp.replace('/*DELT*/', '').split('\t')[1]
                                ds = tmp.replace('/*DELT*/', '').split('\t')[2]
                                dic_settings[item].append([(th, op), 'DELT', ds])
                            elif '/*UNIT*/' in tmp:
                                th = tmp.replace('/*UNIT*/', '').split('\t')[0]
                                op = tmp.replace('/*UNIT*/', '').split('\t')[1]
                                ds = tmp.replace('/*UNIT*/', '').split('\t')[2]
                                rate = tmp.replace('/*UNIT*/', '').split('\t')[3]
                                dic_settings[item].append([(th, op), 'UNIT', ds, rate])

    return dic_settings


# generate bound settings
def read_filter_bound_setting(setting_file_path):
    dic_bound = {}
    with open(setting_file_path, 'r', encoding='utf-8') as f_conf:
        lines = f_conf.readlines()
        op = ''
        for line in lines:
            tmp = line.strip()
            if len(tmp) != 0:
                if '/*MEMO*/' in tmp:
                    continue
                else:
                    if '/*OP*/' in tmp:
                        op = tmp.replace('/*OP*/', '')
                        dic_bound[op] = []
                    else:
                        if '/*UP*/' in tmp:
                            up = float(tmp.replace('/*UP*/', ''))
                            dic_bound[op].append(up)
                        elif '/*DOWN*/' in tmp:
                            dn = float(tmp.replace('/*DOWN*/', ''))
                            dic_bound[op].append(dn)
    return dic_bound


# load dictionary dict[thing] - (title,)
def read_thing_title_setting(setting_file_path):
    dic_th_title = {}
    with open(setting_file_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            tmp = line.strip()
            if '(\'' in tmp:
                res = ast.literal_eval(tmp)
            else:
                dic_th_title[tmp] = res
    return dic_th_title
