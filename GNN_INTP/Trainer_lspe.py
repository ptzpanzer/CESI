from config_manager import myconfig

import os
import sys
import json
import time
from datetime import datetime

import solver_lspe as solver
import config_manager
import support_functions


if __name__ == '__main__':
    settings = json.loads(sys.argv[3])
    settings["job_id"] = sys.argv[1]
    settings["tmpdir"] = sys.argv[2]

    # solver = config_manager.load_solver(f"solver_{settings['model'].lower()}")
    
    # build working folder
    result_sheet = []
    list_total, list_err = solver.training(settings=settings)
    mae, mape, rsq = solver.evaluate(settings=settings)
    result_sheet.append([list_total, list_err, mae, mape, rsq])
    
    # collect wandb result into file
    rtn = {
        "MAE": sum(result_sheet[0][2])/len(result_sheet[0][2]), 
        "R2": sum(result_sheet[0][4])/len(result_sheet[0][4]), 
        "MAPE": sum(result_sheet[0][3])/len(result_sheet[0][3]), 
        "list_total_0": result_sheet[0][0],
        "list_err_0": result_sheet[0][1],
    }
    json_dump = json.dumps(rtn)
    with open(settings['coffer_slot'] + f"/{settings['job_id']}.rtn", 'w') as fresult:
        fresult.write(json_dump)
    