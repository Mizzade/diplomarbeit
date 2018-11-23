#!/usr/bin/env python3
from typing import List, Tuple, Any
import subprocess
import os
import sys
import json
import argparse
import pickle
import config_eval as ce

def get_file_list(data_dir: str, allowed_extensions: List[str]) -> List[str]:
    file_list = []
    for image_set in next(os.walk(data_dir))[1]:
        image_set_path = os.path.join(data_dir, image_set)
        for file in os.listdir(image_set_path):
            _, f_ext = os.path.splitext(file)
            if f_ext.lower() in allowed_extensions:
                file_list.append(os.path.join(image_set_path, file))

    return file_list

def run_network(network: Any, config: argparse.Namespace):
    if network.name == 'TILDE':
        return subprocess.check_call(['/bin/bash', './{}'.format(network.main),
            config.data_dir, config.output_dir, json.dumps(file_list)],
            cwd=network.dir)
    else:
        return subprocess.check_call(['pipenv', 'run', 'python',
        './{}'.format(network.main), json.dumps(dict(network._asdict())),
        json.dumps(vars(config)), json.dumps(file_list)],
        cwd=network.dir)

def run_network2(path: str, name: str, main: str, output_dir: str, file_list: List[str], root_dir: str, data_dir: str, **kwargs) -> List[Tuple[Any]]:
    if name == 'TILDE':
        return subprocess.check_call(['/bin/bash', './{}'.format(main), data_dir, output_dir,
        json.dumps(file_list)], cwd=path)
    else:
        return subprocess.check_call(['pipenv', 'run', 'python', './{}'.format(main),
            output_dir, json.dumps(file_list)], cwd=path)

if __name__ == "__main__":
    argv = sys.argv[1:]
    config, networks = ce.get_config(argv)
    file_list = sorted(get_file_list(config.data_dir, config.allowed_extensions))

    if config.max_num_images is not None:
        file_list = file_list[:config.max_num_images]

    for n in networks:
        print('Starting network `{}`.'.format(n.name))
        _ = run_network(n, config)
        print('Network `{}` done.\n'.format(n.name))
