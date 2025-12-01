from config_manager import myconfig

import wandb
import subprocess
import os
import json
import time
from datetime import datetime

import support_functions


def get_hpcheader(full_setting):    
    coffer_slot = full_setting['coffer_slot']
    model = full_setting['model']
    dataset = full_setting['dataset'] + "_" + full_setting['postfix']
    holdout = full_setting['holdout']

    if (model == 'GCN' and dataset == 'ABO_res250_reg4c') or model in myconfig.m_GPU_models:
        hpc_header = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig.project_name}
#SBATCH --error={coffer_slot}%x.%j.err
#SBATCH --output={coffer_slot}%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={myconfig.e_mail}
#SBATCH --export=ALL

#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID

mkdir -p $TMPDIR/{dataset}/pkl/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_train_-1/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_test_0/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_0/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_20/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_50/
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_train_-1/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_train_-1.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_test_0/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_test_0.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_0/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_0.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_20/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_20.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_50/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_50.tar.gz
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

#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID

mkdir -p $TMPDIR/{dataset}/pkl/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_train_-1/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_test_0/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_0/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_20/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_50/
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_train_-1/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_train_-1.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_test_0/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_test_0.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_0/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_0.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_20/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_20.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_50/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_50.tar.gz
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

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=180000mb
#SBATCH --time=24:00:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID

mkdir -p $TMPDIR/{dataset}/pkl/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_train_-1/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_test_0/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_0/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_20/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_50/
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_train_-1/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_train_-1.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_test_0/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_test_0.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_0/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_0.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_20/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_20.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_50/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_50.tar.gz
"""
        return hpc_header

    elif model in myconfig.dev_GPU_models:
        hpc_header = \
f"""#!/bin/bash
#SBATCH --job-name={myconfig.project_name}
#SBATCH --error={coffer_slot}%x.%j.err
#SBATCH --output={coffer_slot}%x.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={myconfig.e_mail}
#SBATCH --export=ALL

#SBATCH --partition=dev_gpu_4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

eval \"$(conda shell.bash hook)\"
conda activate {myconfig.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{myconfig.conda_env}/lib

job_id=$SLURM_JOB_ID

mkdir -p $TMPDIR/{dataset}/pkl/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_train_-1/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_test_0/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_0/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_20/
mkdir -p $TMPDIR/{dataset}/pkl/4_{holdout}_eval_50/
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_train_-1/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_train_-1.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_test_0/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_test_0.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_0/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_0.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_20/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_20.tar.gz
tar -C $TMPDIR/{dataset}/pkl/4_{holdout}_eval_50/ -xzf $(ws_find blah)/{dataset}/pkl/4_{holdout}_eval_50.tar.gz
"""
        return hpc_header
        
    
    
    


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

            if full_setting["model"] == "LSPE2":
                full_setting["postfix"] = "32_5_sep"

            # replace the setting template items that relies on dataset information
            with open(f"./Datasets/{wandb_config['dataset']}/meta_data.json", "r") as f:
                dataset_info = json.load(f)
            
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
            job_script = f"{get_hpcheader(full_setting)}\npython {myconfig.train_script_name} $job_id $TMPDIR '{json.dumps(full_setting)}'"
    
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
