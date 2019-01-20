#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any
import subprocess
import os
import sys
import json
import argparse
import pickle
import config_run_descriptors as ced
import run_support_functions as rsf


def start_subprocess(config:Dict, file_list:List[str]) -> None:
    # Build config file
    config_file = '{}_config.pkl'.format(config['descriptor_name'])
    path_to_config_file = os.path.join(config['tmp_dir_{}'.format(config['descriptor_name'])], config_file)

    # Create tmp dir if it not exists:
    rsf.create_dir(config['tmp_dir_{}'.format(config['descriptor_name'])])

    # Write config file into tmp dir:
    rsf.write_config_file(path_to_config_file, [config, file_list])

    try:
        status_code = subprocess.check_call(['pipenv', 'run', 'python',
                './{}'.format(config['main_{}'.format(config['descriptor_name'])]),
                path_to_config_file],
                cwd=config['root_dir_{}'.format(config['descriptor_name'])])
    except Exception as e :
        print('Error in subprocess call for descriptor {} and detector {}.'.format(config['descriptor_name'], config['detector_name']))
        print('Reason:\n', e)

    # Remove tmp dir
    rsf.remove_dir(config['tmp_dir_{}'.format(config['descriptor_name'])])

def main(config):
    file_list = rsf.get_file_list(config)

    for descriptor_name in config['descriptors']:
        for detector_name in config['detectors']:
            print('\nStarting descriptor `{}` with detector `{}`'.format(descriptor_name, detector_name))
            config['detector_name'] = detector_name
            config['descriptor_name'] = descriptor_name
            start_subprocess(config, file_list)
            print(('Descriptor `{}` with detector `{}` done.'.format(descriptor_name, detector_name)))


if __name__ == "__main__":
    argv = sys.argv[1:]
    config = ced.get_config(argv)

    if config['dry']:
        rsf.print_configuration(config)
    else:
        main(config)
