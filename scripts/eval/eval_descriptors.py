#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any
import subprocess
import os
import sys
import json
import argparse
import pickle
import config_eval_descriptors as ced
import eval_support_functions2 as esf
import multiprocessing as mp
import copy

def start_subprocess(config:Dict, file_system:Dict) -> None:
    eval_dir = os.path.join(
        config['root_dir'],
        config['root_dir_evaluation'])

    tmp_dir = os.path.join(
        eval_dir,
        config['tmp_dir_evaluation'])

    # Build config file
    config_file = '{}_{}_config.pkl'.format(config['descriptor_name'], config['detector_name'])
    path_to_config_file = os.path.join(tmp_dir, config_file)

    # Build config file
    path_to_config_file = os.path.join(config['tmp_dir_descriptor'], config_file)

    # Write config file into tmp dir:
    # Write config file into tmp dir:
    esf.write_config_file(path_to_config_file, [config])

    try:
        subprocess.check_call(['pipenv', 'run', 'python', 'run_eval2.py',
            path_to_config_file],
            cwd=eval_dir)

    except Exception as e:
        print('Could not evaluate descriptor {} with detector {}'.format(config['descriptor_name'], config['detector_name']))
        print('Reason:\n', e)

def main(config:Dict) -> None:

    tmp_dir = os.path.join(
        config['root_dir'],
        config['root_dir_evaluation']
        config['tmp_dir_evaluation'])

    # Setup a list of processes to be run
    processes = []

    for descriptor_name in config['descriptors']:
        for detector_name in config['detectors']:
            _config = copy.deepcopy(config)
            _config['descriptor_name'] = descriptor_name
            _config['detector_name'] = detector_name

        processes.append(mp.Process(
            target=start_subprocess,
            args=(_config,)))

    # Create tmp dir if it not exists:
    esf.create_dir(tmp_dir)

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Remove tmp dir
    esf.remove_dir(tmp_dir)

if __name__ == "__main__":
    argv = sys.argv[1:]
    config = ced.get_config(argv)

    if config['dry']:
        esf.print_configuration(config)
    else:
        main(config)
