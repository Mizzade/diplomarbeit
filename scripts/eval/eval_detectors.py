#!/usr/bin/env python3
from typing import List, Tuple, Dict, Any
import subprocess
import os
import sys
import json
import argparse
import pickle
import config_eval_detectors as ced
import eval_support_functions as esf
import multiprocessing as mp
import copy

def start_subprocess(config:Dict, file_system:Dict) -> None:
    # Build config file
    config_file = '{}_config.pkl'.format(config['detector_name'])
    path_to_config_file = os.path.join(config['tmp_dir_detector'], config_file)

    # Write config file into tmp dir:
    esf.write_config_file(path_to_config_file, [config, file_system])

    try:
        subprocess.check_call(['pipenv', 'run', 'python', 'run_eval.py',
            path_to_config_file],
            cwd=config['root_dir_detector'])

    except Exception as e:
        print('Could not evaluate detector {}'.format(config['detector_name']))
        print('Reason:\n', e)

def main(config:Dict) -> None:
    # Setup a list of processes to be run
    processes = []

    for detector_name in config['detectors']:
        _config = copy.deepcopy(config)
        _config['detector_name'] = detector_name

        file_system = esf.build_file_system(_config, fs_type='keypoints')
        processes.append(mp.Process(
            target=start_subprocess,
            args=(_config, file_system)))

    # Create tmp dir if it not exists:
    esf.create_dir(config['tmp_dir_detector'])

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Remove tmp dir
    esf.remove_dir(config['tmp_dir_detector'])

if __name__ == "__main__":
    argv = sys.argv[1:]
    config = ced.get_config(argv)

    if config['dry']:
        esf.print_configuration(config)
    else:
        main(config)
