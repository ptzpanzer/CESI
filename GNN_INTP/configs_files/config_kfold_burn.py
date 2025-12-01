# flex project settings
#  - project name on wandb and HPC
project_name = 'FE_burn'
#  - project discription
project_dscr = "FEGNN final stage"
# is it a new run or continue
debug = False
new_run = True
new_mode = ""
sweep_id = ''

# fixed project settings
#  - wandb api key
api_key = ''
#  - entity name (your wandb account name)
entity_name = ''
#  - e-mail address to recieve notifications
e_mail = ''
#  - conda location
conda_env = ''
#  - file name of the slurm_wrapper, don't change this if you haven't write a new one
slurm_wrapper_name = './burn_wrapper.py'
#  - file name of the training code
train_script_name = './burn_Trainer.py'

# fixed working dirs
# where should the intermedia generated scripts be saved (automatically cleaned at the start of each run)
slurm_scripts_path = f'./{project_name}_slurm_scripts/'
# where should the outputs & logs be saved (automatically cleaned at the start of each run)
log_path = f'./{project_name}_logs/'
# where should calculation nodes save their important results (e.g. best model weights)
coffer_path = f'./{project_name}_coffer/'


# Sweep definition
#     - how many sweeps do you want to run parallelly
pool_size = 48

sweep_config = {
    "project": project_name,
    'program': slurm_wrapper_name,
    "name": "offline-sweep",
    
    'method': 'grid',

    "metric": {
        'name': 'best_err',
        'goal': 'minimize',   
    },
    
    'parameters':{
        "model": {
            'values': ["TRSAGE5_NA", "TRSAGE5", "Transformer_NA", "PEGSAGE", ]
        },
        "seed": {
            'values': [1, 2, 3, ]
        },
        
        "dataset": {
            'values': [
                "OceanAt_res250_reg4c", 
            ]
        },
        "fold": {
            'values': [4, ]
        },
        "holdout": {
            'values': [3, ]
        },

        "burn_rate": {
            'values': [0.2, 0.4, 0.6, 0.8]
        },
    },
}

CPU_models = ["GCN", "GSAGE", "KSAGE", "PEGSAGE", ]
m_GPU_models = ["GAT", ]
s_GPU_models = ["TRSAGE5", "TRSAGE5_NA", "TRSAGE5_NKL", "TRSAGE5_ND", "Transformer_NA", ]

