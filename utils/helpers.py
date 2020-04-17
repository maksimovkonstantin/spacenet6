import argparse
import os.path as osp
import sys
from importlib import import_module


def get_config(filename):
    module_name = osp.basename(filename)[:-3]
    
    config_dir = osp.dirname(filename)
    
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
    return cfg_dict


def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args
