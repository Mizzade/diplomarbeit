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

def detect(
    image_path: str,
    config: dict) -> None:
    """Detects keypoints for a given input image.
    Draws keypoints into the image.
    Returns keypoints, heatmap and image with keypoints.

    Arguments:
        image_path {str} -- Path to the image.
        config {dict} -- General configuations. See config_run_detectors.py.

    Returns:
        Tuple[List[cv2.KeyPoint], np.array, (np.array | None) ] -- Returns list
        of cv2.KeyPoint, an image with the corresponding keypoints, and if
        available, an heatmap.

    1) Create temporary folder `tmp` to save intermediate output.
    2a) Load and smart scale the image
    2b) Save the resulting image in `tmp`.
    3a) Subprocess call to TILDE for keypoints. Save output in `tmp`
    4a) Load keypoints from 'tmp' and convert keypoints to cv2.Keypoints.
    4b) Draw list of cv2.KeyPoints into image.
    5) Return KeyPoint list and image with keypoints.
    """

    # 1)
    io_utils.create_dir(config['tmp_dir_tilde'])

    # 2)
    img = cv2.imread(image_path)
    img = io_utils.smart_scale(img, config['max_size'], prevent_upscaling=True) if config['max_size'] is not None else img

    # 2b)
    tmp_filename = 'tmp_img.png'
    tmp_keypoints = 'keypoints.csv'
    tmp_heatmap = 'heatmap.csv'

    path_tmp_img = os.path.join(config['tmp_dir_tilde'], tmp_filename)
    path_tmp_kpts = os.path.join(config['tmp_dir_tilde'], tmp_keypoints)
    path_tmp_heatmap = os.path.join(config['tmp_dir_tilde'], tmp_heatmap)

    cv2.imwrite(path_tmp_img, img)

    # 3a)
    imageDir = config['tmp_dir_tilde']
    outputDir = config['tmp_dir_tilde']
    fileName = tmp_filename
    filterPath = '/home/tilde/TILDE/c++/Lib/filters'
    filterName = 'Mexico.txt'

    # Call use_tilde.cpp
    # The output will be saved into
    # - config['tmp_dir_tilde']/keypoints.csv and
    # - config['tmp_dir_tilde']/heatmap.csv
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

    return (kpts, img_kp, heatmap)

def main(argv: Tuple[str]) -> None:
    """Runs the TILDE model and saves the results.

    Arguments:
        argv {Tuple[str]} -- List of one parameters. There should be exactly
            one parameter - the path to the config file inside the tmp dir.
            This config file will be used to get all other information and
            process the correct images.
    """
    if len(argv) <= 0:
        raise RuntimeError("Missing argument <path_to_config_file>. Abort")

    with open(argv[0], 'rb') as src:
        config_file = pickle.load(src, encoding='utf-8')

    detector_name, config, file_list = config_file

    for file in tqdm(file_list):
        keypoints, keypoints_image, heatmap_image = detect(file, config)
        io_utils.save_detector_output(file, detector_name, config, keypoints,
            keypoints_image, heatmap_image)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
