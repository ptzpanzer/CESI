from config_manager import myconfig

import wandb
import yaml
import os
from multiprocessing import Process
import subprocess

import support_functions


# define & save the yaml file for wandb sweep
def make_para_yaml(project_name, slurm_wrapper, scripts_path, sweep_config):
    # make the yaml file
    with open(scripts_path + 'sweep_params.yaml', 'w') as outfile:
        yaml.dump(sweep_config, outfile, default_flow_style=False)


if __name__ == "__main__":
    # prepare working folders
    if myconfig.new_run:
        if myconfig.new_mode == "clean":
            support_functions.build_folder_and_clean('./wandb/')
            support_functions.build_folder_and_clean(myconfig.slurm_scripts_path)
            support_functions.build_folder_and_clean(myconfig.log_path)
            support_functions.build_folder_and_clean(myconfig.coffer_path)
        elif myconfig.new_mode == "archive":
            support_functions.build_folder_and_archive('./wandb/')
            support_functions.build_folder_and_archive(myconfig.slurm_scripts_path)
            support_functions.build_folder_and_archive(myconfig.log_path)
            support_functions.build_folder_and_archive(myconfig.coffer_path)
    
    # login to wandb
    wandb.login(key=myconfig.api_key)
    if myconfig.new_run:
        # generate & load sweep config
        make_para_yaml(myconfig.project_name, myconfig.slurm_wrapper_name, myconfig.slurm_scripts_path, myconfig.sweep_config)
        with open(myconfig.slurm_scripts_path + 'sweep_params.yaml') as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        # start sweep
        sweep_id = wandb.sweep(sweep=config_dict)
    else:
        sweep_id = myconfig.sweep_id
        
    while True:
        status = support_functions.get_sweep_status(entity=myconfig.entity_name, project=myconfig.project_name, sweep_id=sweep_id)
        if status == "FINISHED":
            print("All task delivered.")
            break
        wandb.agent(sweep_id=sweep_id, entity=myconfig.entity_name, project=myconfig.project_name, count=1)
