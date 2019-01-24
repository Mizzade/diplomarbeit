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

def start_subprocess(config:Dict, file_list:List[str]) -> None:
    # Build config file
    config_file = '{}_config.pkl'.format(config['detector_name'])
    path_to_config_file = os.path.join(config['tmp_dir_{}'.format(config['detector_name'])], config_file)

    # Create tmp dir if it not exists:
    rsf.create_dir(config['tmp_dir_{}'.format(config['detector_name'])])

    # Write config file into tmp dir:
    rsf.write_config_file(path_to_config_file, [config, file_list])


    if config['detector_name'] == 'tilde':
        try:
            # Call shell script to call docker container.
            status_code = subprocess.check_call(['/bin/bash', './{}'.format(config['main_{}'.format(config['detector_name'])]),
                config['data_dir'],
                config['output_dir'],
                config['tmp_dir_tilde'],
                path_to_config_file],
                cwd=config['root_dir_{}'.format(config['detector_name'])])

        except Exception as e:
            print('Error in subprocess call for detector {}.'.format(config['detector_name']))
            print('Reason:\n', e)
            # You have to give the rights back to USER, since Docker writes to root
            # sudo chown -R $USER outputs
        env = os.environ.copy()
        subprocess.check_call(['sudo', 'chown', '-R', env['USER'], config['output_dir']])

    else:
        try:
            status_code = subprocess.check_call(['pipenv', 'run', 'python',
                './{}'.format(config['main_{}'.format(config['detector_name'])]),
                path_to_config_file],
                cwd=config['root_dir_{}'.format(config['detector_name'])])

        except Exception as e:
            print('Error in subprocess call for detector {}.'.format(config['detector_name']))
            print('Reason:\n', e)

    # Remove tmp dir
    rsf.remove_dir(config['tmp_dir_{}'.format(config['detector_name'])])

def main(config):
    file_list = rsf.get_file_list(config)

    for detector_name in config['detectors']:
        print('\nStarting detector `{}`'.format(detector_name))
        config['detector_name'] = detector_name
        config['descriptor_name'] = None
        start_subprocess(config, file_list)
        print(('Detector `{}` done.'.format(detector_name)))


if __name__ == "__main__":
    argv = sys.argv[1:]
    config = ced.get_config(argv)

    if config['dry']:
        rsf.print_configuration(config)
    else:
        main(config)
