#!/usr/bin/env python3
from typing import List, Tuple, Any
import subprocess
import os
import sys
import json
import argparse
import pickle

networks = [
    {
        'name': 'SIFT',
        'dir': 'pipe_sift',
        'main': 'use_sift.py'
    },
    # {
    #     'name': 'SuperPoint',
    #     'dir': 'pipe_superpoint',
    #     'main': 'use_superpoint.py'
    # },
    # {
    #     'name': 'Tfeat',
    #     'dir': 'desc_tfeat',
    #     'main': 'use_tfeat.py'
    # }
]

allowed_extensions = ['.jpg', '.png', '.ppm', '.jpeg', '.tiff']

def get_file_list(data_dir: str) -> List[str]:
    file_list = []
    for image_set in next(os.walk(data_dir))[1]:
        image_set_path = os.path.join(data_dir, image_set)
        for file in os.listdir(image_set_path):
            _, f_ext = os.path.splitext(file)
            if f_ext.lower() in allowed_extensions:
                file_list.append(os.path.join(image_set_path, file))

    return file_list

def run_network(path: str, name: str, main: str, output_dir: str, file_list: List[str], **kwargs) -> List[Tuple[Any]]:
    return subprocess.check_call(['pipenv', 'run', 'python', './{}'.format(main),
        output_dir, json.dumps(file_list)], cwd=path)

if __name__ == "__main__":
    argv = sys.argv[1:]

    # First and only argument must be root dir of project
    # TODO catch argv errors
    if len(argv) < 0:
        raise RuntimeError("Missing argument root path. Abort.")

    root_dir = argv[0]
    data_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'outputs')
    file_list = get_file_list(data_dir)

    for n in networks:
        print('Starting network `{}`.'.format(n['name']))
        network_dir = os.path.join(root_dir, n['dir'])
        status_code = run_network(network_dir, **n, output_dir=output_dir, file_list=file_list)
        print('Network `{}` done.\n'.format(n['name']))
