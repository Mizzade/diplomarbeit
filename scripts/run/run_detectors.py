#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any
import subprocess
import os
import sys
import json
import argparse
import pickle
import shutil
import config_run_detectors as ced
import run_support_functions as rsf

def start_subprocess_for_model(model:str, config:Dict, file_list:List[str]) -> None:
    if model == 'tilde':
        config_file = 'tilde_config.pkl'
        path_to_config_file = os.path.join(config['tmp_dir_tilde'], config_file)

        # Create tmp dir if it not exists:
        if not os.path.exists(config['tmp_dir_tilde']):
            os.makedirs(config['tmp_dir_tilde'], exist_ok=True)

        # Write config file into tmp dir:
        with open(path_to_config_file, 'wb') as dst:
            pickle.dump(
                [model, config, file_list],
                dst,
                protocol=pickle.HIGHEST_PROTOCOL)

        # Call shell script to call docker container.
        status_code = subprocess.check_call(['/bin/bash', './{}'.format('use_tilde.sh'),
            config['data_dir'],
            config['output_dir'],
            config['tmp_dir_tilde'],
            path_to_config_file],
            cwd=config['root_dir_tilde'])

        # Remove tmp dir
        if os.path.exists(config['tmp_dir_tilde']):
            shutil.rmtree(config['tmp_dir_tilde'], ignore_errors=True)

        return status_code

    else:
        # TODO: Handle tcovdet later.
        pass

def main(config):
    file_list = rsf.get_file_list(config)

    for d in config['detectors']:
        print('\nStarting detector `{}`'.format(d))
        start_subprocess_for_model(d, config, file_list)
        print(('Model `{}` done.'.format(d)))


if __name__ == "__main__":
    argv = sys.argv[1:]
    config = ced.get_config(argv)

    if config['dry']:
        rsf.print_configuration(config)
    else:
        main(config)
