#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any
import subprocess
import os
import sys
import json
import argparse
import pickle
import config_run_detectors as ced
import run_support_functions as rsf

def start_subprocess_for_model(model:str, config:Dict, file_list:List[str]) -> None:
    # Build config file
    config_file = '{}_config.pkl'.format(model)
    path_to_config_file = os.path.join(config['tmp_dir_{}'.format(model)], config_file)

    # Create tmp dir if it not exists:
    rsf.create_dir(config['tmp_dir_{}'.format(model)])

    # Write config file into tmp dir:
    rsf.write_config_file(path_to_config_file, [model, config, file_list])

    if model == 'tilde':
        # Call shell script to call docker container.
        status_code = subprocess.check_call(['/bin/bash', './{}'.format(config['main_{}'.format(model)]),
            config['data_dir'],
            config['output_dir'],
            config['tmp_dir_tilde'],
            path_to_config_file],
            cwd=config['root_dir_{}'.format(model)])

        # You have to give the rights back to USER, since Docker writes to root
        # sudo chown -R $USER outputs
        env = os.environ.copy()
        subprocess.check_call(['sudo', 'chown', '-R', env['USER'], config['output_dir']])

    else:
        status_code = subprocess.check_call(['pipenv', 'run', 'python',
            './{}'.format(config['main_{}'.format(model)]),
            path_to_config_file],
            cwd=config['root_dir_{}'.format(model)])


    # Remove tmp dir
    rsf.remove_dir(config['tmp_dir_{}'.format(model)])

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
