from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import sys
import os
from tqdm import tqdm
import io_utils
import pickle
import subprocess

def detect(
    image_path:str,
    config:Dict
) -> Tuple[List[cv2.KeyPoint], np.array, np.array]:
    """Detects keypoints for a given input image. Draws keypoints in the image.
    Returns keypoints as list of cv2.KeyPoint elements, the input image with
    the keypoints drawn into it aswell as a heatmap, if available.

    Arguments:
        image_path {str} -- Absolute path to the image.
        config {Dict} -- General configuration object. See config_run_detectors.py
        for more information.

    Returns:
        Tuple[List[cv2.KeyPoint], np.array, np.array] -- Keypoints, image with
        kepoints drawn into it and if available, a heatmap (otherwise None).
    """

    """
    1) Create temporary folder `tmp` to save intermediate output.
    2a) Load and smart scale the image
    2b) Save the resulting image in `tmp`.
    3a) Subprocess call to TILDE for keypoints. Save output in `tmp`
    4a) Load keypoints from 'tmp' and convert keypoints to cv2.Keypoints.
    4b) Draw list of cv2.KeyPoints into image.
    5) Return KeyPoint list and image with keypoints.
    """
    raise NotImplemented
    # TODO


def detect_bulk(
    file_list: List[str],
    config:Dict
) -> None:
    """Computes keypoints for all files in `file_list`. Additionally for each
    file, create an image with the corresponding keypoints drawn into it.
    All results will be saved within the `tmp` folder for this module.

    Arguments:
        file_list {List[str]} -- List of all images for which to compute keypoints.
        config {Dict} -- General configuration object. See config_run_detectors.py
        for more information.

    Returns:
        None -- All results here are saved within the `tmp` dir specified within
        the `config` object.
    """

    """
    2) For each file in `file_list` get the corresponding collection and set name
    3) For each collection create a

    """
    # 2)
    file_collection_names = []
    file_set_names = []
    for file_path in file_list:
        collection_name, set_name, _ ,_ = io_utils.get_path_components(file_path)
        file_collection_names.append(collection_name)
        file_set_names.append(set_name)

    # 3)
    unique_collection_names = list(set(file_collection_names))
    unique_set_names = list(set(file_set_names))

    # print('TESTING')
    # for file, collection_name, set_name in zip(file_list, file_collection_names, file_set_names):
    #     print('{}: {} - {}'.format(file, collection_name, set_name))


    # subprocess.check_call(['python', 'patch_network_point_test.py',
    #     '--save_feature', 'covariant_points',
    #     '--output_dir', config['tmp_dir_tcovdet'],
    #     '--'])
    print('Start trying')
    try:
        subprocess.check_call(['python', 'patch_network_point_test.py',
        '--save_feature', 'covariant_points',
        '--output_dir', config['tmp_dir_tcovdet'],
        '--file_list', ' '.join(file_list),
        '--dry'])
    except Exception as e:
        print('Something went terrible wrong.')
        print(e)






def main(argv: Tuple[str]) -> None:
    """Runs the TCovDet model and saves the results.

    Arguments:
        argv {Tuple[str]} -- List of one parameter. There should be exactly
            one parameter - the path to the config file inside the tmp dir.
            This config file will be used to get all other information and
            process the correct images.
    """
    if len(argv) <= 0:
        raise RuntimeError("Missing argument <path_to_config_file>. Abort")

    with open(argv[0], 'rb') as src:
        config_file = pickle.load(src, encoding='utf-8')

    config, file_list = config_file

    if config['bulk_mode_tcovdet']:
        detect_bulk(file_list, config)
        # TODO Save intermediate output as .csv in global output folder
    else:
        raise NotImplemented
        # for file in tqdm(file_list):
        #     keypoints, keypoints_image, heatmap_image = detect(file, config)
        #     io_utils.save_detector_output(file, config['detector_name'], config, keypoints,
        #         keypoints_image, heatmap_image)

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
