#!/usr/bin/env python3
import sys
import subprocess
import typing
import os
import json
from tqdm import tqdm
import pickle
import io_utils
import cv2
from typing import Tuple, List, Any
import numpy as np

def compute(
    model: Any,
    image: str,
    network: dict,
    config: dict) -> None:
    """Computes the keypoints for a given input image.
    Draws keypoints into the image.
    Returns keypoints, heatmap and image with keypoints.

    Arguments:
        model {None} -- A model to load in python.
        image {np.array} -- Path to the image.
        network {dict} -- Config for this network. See config_eval.py.
        config {dict} -- General configuations. See config_eval.py.

    Returns:
        Tuple[List[cv2.KeyPoint], None, np.array, np.array] -- Returns tuple (keypoints, None, image with keypoints, image of heatmap).

    1) Create temporary folder `tmp` to save intermediate output.
    2a) Load and smart scale the image
    2b) Save the resulting image in `tmp`.
    3a) Subprocess call to TILDE for keypoints. Save output in `tmp`
    4a) Load keypoints from 'tmp' and convert keypoints to cv2.Keypoints.
    4b) Draw list of cv2.KeyPoints into image.
    5) Return KeyPoint list and image with keypoints.
    """

    # 1)
    if not os.path.exists(network['tmp_dir']):
        os.makedirs(network['tmp_dir'], exist_ok=True)

    # 2)
    img = cv2.imread(image)
    img = io_utils.smart_scale(img, config['size'], prevent_upscaling=True) if config['size'] is not None else img

    # 2b)
    tmp_filename = 'tmp_img.png'
    tmp_keypoints = 'keypoints.csv'
    tmp_heatmap = 'heatmap.csv'

    path_tmp_img = os.path.join(network['tmp_dir'], tmp_filename)
    path_tmp_kpts = os.path.join(network['tmp_dir'], tmp_keypoints)
    path_tmp_heatmap = os.path.join(network['tmp_dir'], tmp_heatmap)

    cv2.imwrite(path_tmp_img, img)

    # 3a)
    imageDir = network['tmp_dir']
    outputDir = network['tmp_dir']
    fileName = tmp_filename
    filterPath = '/home/tilde/TILDE/c++/Lib/filters'
    filterName = 'Mexico.txt'

    # Call use_tilde.cpp
    # The output will be saved into
    # - config['tmp_dir']/keypoints.csv and
    # - config['tmp_dir']/heatmap.csv
    subprocess.check_call([
        './use_tilde',
        '--imageDir', imageDir,
        '--outputDir', outputDir,
        '--fileName', fileName,
        '--filterPath', filterPath,
        '--filterName', filterName])

    # 4)
    kpts_file = np.loadtxt(path_tmp_kpts, dtype=int, comments='#', delimiter=', ')
    kpts = [cv2.KeyPoint(x[0], x[1], _size=1) for x in kpts_file]
    heatmap = np.loadtxt(path_tmp_heatmap, dtype=float, comments='# ', delimiter=', ')
    img_kp = cv2.drawKeypoints(img, kpts, None)

    return (kpts, None, img_kp, heatmap)

def main(argv: Tuple[str]) -> None:
    """Runs the TILDE model and saves the results.

    Arguments:
        argv {Tuple[str]} -- List of one parameters. There should be exactly
            one paramter - the path to the config file inside the tmp dir.
            This config file will be used to get all other information and
            process the correct images.
    """
    if len(argv) <= 0:
        raise RuntimeError("Missing argument <path_to_config_file>. Abort")

    with open(argv[0], 'rb') as src:
        config_file = pickle.load(src, encoding='utf-8')

    network, config, file_list = config_file
    model = None

    project_name = 'tilde'
    detector_name = network['name']
    descriptor_name = None

    for file in tqdm(file_list):
        io_utils.save_output(
            file,
            compute(model, file, network, config),
            config['output_dir'],
            detector_name,
            descriptor_name,
            project_name)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
