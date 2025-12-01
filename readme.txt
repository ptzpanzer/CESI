This is the code appendix to PAKDD 2025 Paper: 
Li, Chaofan, Till Riedel, and Michael Beigl. "CESI: Sparse Input Spatial Interpolation for Heterogeneous and Noisy Hybrid Wireless Sensor Networks." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Cham: Springer Nature Switzerland, 2025.


1. First, download dataset and transform them to the 'Standard Format' with the codes in './Data_Preprocessing/', further instructions see readme.txt in the folders.
	- As a result of this step, each dataset will become a set of narrow format .csv files stored in a Folder named 'Dataset_name/Dataset_Separation/'

2. Put datasets in standard format from step 1 into existing folders in './GNN_INTP/Datasets/'
	- When set correctly, the path will be like './GNN_INTP/Datasets/ABO_res250/Dataset_Separation/'
	- 'OceanAt' refers also to the Marine dataset

3. Run 'convert_to_region4.py' in './GNN_INTP/Datasets/', this will further preprocessing the data
	- You will get 'ABO_res250_reg4c', 'OceanAt_res250_reg4c' and 'SAQN_res250_reg4c'
	- Copy json files in './GNN_INTP/Datasets/metas/' into those folders according to the file name, and rename them all as 'meta_data.json'
	- When set correctly, 'ABO_res250_reg4c', 'OceanAt_res250_reg4c' and 'SAQN_res250_reg4c' folders will have following folders and files:
		- A 'Dataset_Separation' folder, which saves dataset .csv files
		- A 'meta_data.json' file, which saves meta data of the dataset
		- A 'log.pkl' file, which saves how different op_names are seperated in different leave-one-area-out fold

4. Run '02.Fold_Divide.py' in './GNN_INTP/Datasets/', this will analyse informations for different folds of leave-one-area-out cross validation
	- When set correctly, 'ABO_res250_reg4c', 'OceanAt_res250_reg4c' and 'SAQN_res250_reg4c' folders will add following folder:
		- A 'Folds_info' folder, which saves information about each fold of 4-fold leave-one-area-out cross validation

5. Run Overall Experiments & Ablation study:
	- 1. Set Experiments configs in './GNN_INTP/configs_files/config_kfold_trans.py'
		We use experiment management software W&B, you need to fill in your own api_key, entity_name, e_mail, and conda_env.
		You also need to set sweep_config to define which models on which datasets you want to carry out the experiment.
		The naming in the code is slightly different as they are in the paper:
			GCN - GCN
			GAT - GAT
			GSAGE - GraphSAGE
			KSAGE - KCN (using GraphSAGE as backbone)
			PEGSAGE - PE-GNN (using GraphSAGE as backbone)
			Transformer_NA - Vanilla Transformer
			TRSAGE5 - CESI
			TRSAGE5_ND - CESI w/o $L_{CC}$
			TRSAGE5_NKL - CESI w/o $L_{KL}$
			TRSAGE5_NA - CESI Null
	- 2. Set use_config = "config_kfold_trans" in './GNN_INTP/config_manager.py'
	- 3. Run './GNN_INTP/01.wandb_start.py'
		We use a slurm based HPC, the task is triggered in line 154 in './GNN_INTP/slurm_wrapper.py' by submitting generated sbatch file to slurm. If you use other running environments, please edit this part by yourself

6. Run Experiments on Robustness:
	- 1. First, we need to prepare some datasets, run './GNN_INTP/Datasets/prepare_burn.py' to do this
		You will get 4 more folders with datasets in './GNN_INTP/Datasets/', namingly 'OceanAt_res250_reg4c_b_0.2' to 'OceanAt_res250_reg4c_b_0.8'
		copy 'meta_data.json', 'log.pkl' and 'Folds_info' in OceanAt_res250_reg4c to all these folders
	- 2. Set Experiments configs in './GNN_INTP/configs_files/config_kfold_burn.py'
	- 3. Set use_config = "config_kfold_burn" in './GNN_INTP/config_manager.py'

	- 4. Run './GNN_INTP/01.wandb_start.py'
