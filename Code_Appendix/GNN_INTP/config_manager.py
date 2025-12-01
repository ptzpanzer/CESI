use_config = "config_kfold_trans"

import importlib


def load_config(config_name):
    try:
        config_module = importlib.import_module(f"configs_files.{config_name}")
        return config_module
    except ImportError:
        raise ImportError(f"Config version '{config_name}' not found")
        

# 默认加载 config.py
myconfig = load_config(use_config)
