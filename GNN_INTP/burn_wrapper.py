from config_manager import myconfig

import wandb
import subprocess
import os
import json
import time
from datetime import datetime

import support_functions


def get_hpcheader(coffer_slot, model, dataset):
    if (model == 'GCN' and dataset == 'ABO_res250_reg4c') or model in myconfig.m_GPU_models:
        hpc_header = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig.project_name}
#SBATCH --error={coffer_slot}%x.%j.err
#SBATCH --output={coffer_slot}%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={myconfig.e_mail}
#SBATCH --export=ALL

#SBATCH --partition=GPU4
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID
"""
        return hpc_header

    elif model in myconfig.s_GPU_models:
        hpc_header = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig.project_name}
#SBATCH --error={coffer_slot}%x.%j.err
#SBATCH --output={coffer_slot}%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={myconfig.e_mail}
#SBATCH --export=ALL

#SBATCH --partition=GPU4
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID
"""
        return hpc_header

    elif model in myconfig.CPU_models:
        hpc_header = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig.project_name}
#SBATCH --error={coffer_slot}%x.%j.err
#SBATCH --output={coffer_slot}%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={myconfig.e_mail}
#SBATCH --export=ALL

#SBATCH --partition=GPU4
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID
"""
        return hpc_header
        
    
def get_run(ds, m, s, h, valid_runs):
    count = 0
    for run in valid_runs:
        if run.config["dataset"] == ds and run.config["model"] == m and run.config["seed"] == s and run.config["holdout"] == h:
            rtn = run
    return rtn


def get_real_path(r, coffer_roots):
    logged_path = r.summary["coffer_slot"].split("/")[-2]

    for root in coffer_roots:
        if logged_path in os.listdir(root):
            return root + logged_path + "/"
    


def wrap_task(config=None):
    with wandb.init(config=config):
        # recieve wandb config for this run from Sweep Controller
        wandb_config = dict(wandb.config)

        model_available = os.path.exists(f"./solver_files/solver_{wandb_config['model'].lower()}.py")
        dataset_available = os.path.exists(f"./Datasets/{wandb_config['dataset']}/meta_data.json")
        fh_available = os.path.exists(f"./Datasets/{wandb_config['dataset']}/Folds_Info/divide_set_{wandb_config['fold']}_{wandb_config['holdout']}.info")

        if not model_available or not dataset_available or not fh_available:
            return f"Unvalid Run: model {model_available}, dataset {dataset_available}, f&h {fh_available}"
        else:
            # get corresponding setting template by model
            full_setting = {}
            
            # replace the setting template items that redefined in wandb config
            full_setting["agent_id"] = wandb.run.id
            full_setting["coffer_slot"] = myconfig.coffer_path + datetime.now().strftime("%Y%m%d%H%M%S") + "-" + wandb.run.id + '/'
            support_functions.make_dir(full_setting['coffer_slot'])
            full_setting["debug"] = myconfig.debug
            full_setting.update(wandb_config)

            # replace the setting template items that relies on dataset information
            with open(f"./Datasets/{wandb_config['dataset']}/meta_data.json", "r") as f:
                dataset_info = json.load(f)

            with open(f"./model_paths.json", "r") as f:
                model_paths = json.load(f)

            key = f"{full_setting['dataset']}_{full_setting['model']}_{full_setting['holdout']}_{full_setting['seed']}"
            full_setting["model_path"] = model_paths[key]

            # wait until available pipe slot
            while True:
                cmd = f"squeue"
                status = subprocess.check_output(cmd, shell=True).decode()
                lines = status.split('\n')[1:-1]
                if len(lines) <= myconfig.pool_size:
                    break
                else:
                    time.sleep(60)
    
            # then build up the slurm script
            job_script = f"{get_hpcheader(full_setting['coffer_slot'], full_setting['model'], full_setting['dataset'])}\npython {myconfig.train_script_name} $job_id '{json.dumps(full_setting)}'"
    
            # Write job submission script to a file
            with open(myconfig.slurm_scripts_path + f"{wandb.run.id}.sbatch", "w") as f:
                f.write(job_script)
    
            # Submit job to Slurm system and get job ID
            cmd = "sbatch " + myconfig.slurm_scripts_path + f"{wandb.run.id}.sbatch"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            job_id = output.split()[-1]
            wandb.log({
                "job_id" : job_id,
                "coffer_slot": full_setting["coffer_slot"]
            })
            return job_id
        
           
if __name__ == '__main__':
    rtn = wrap_task()
    print(f'\n***********Process Finished with: {rtn}***********\n')
    wandb.finish()
