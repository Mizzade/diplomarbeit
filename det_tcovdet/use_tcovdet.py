from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import sys
import os
from tqdm import tqdm
import io_utils
import pickle
import subprocess
from scipy.io import savemat, loadmat

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

    try:
        # Create feature map for each image in 'covariant_points' folder
        subprocess.check_call(['python', 'patch_network_point_test.py',
        '--save_feature', 'covariant_points',
        '--output_dir', config['tmp_dir_tcovdet'],
        '--file_list', ' '.join(file_list),
        ])

    except Exception as e:
        print('TCovDet: Covariant feature map creation failed.')
        print(e)
        raise e

    try:
        collection_names = []
        set_names = []
        image_names = []
        for file_path in file_list:
            collection_name, set_name, file_base, extension = io_utils.get_path_components(file_path)
            collection_names.append(collection_name)
            set_names.append(set_name)
            image_names.append(file_base)

        # The path to this file
        matlab_config_path = os.path.join(config['tmp_dir_tcovdet'], 'filelist.mat')
        # Where to save the keypoints
        dir_output = os.path.join(config['tmp_dir_tcovdet'], 'feature_points')
        io_utils.create_dir(dir_output) # Create on the fly if not existent

        # Where the .mat files of the covariant step lie
        dir_data = os.path.join(config['tmp_dir_tcovdet'], 'covariant_points')

        # Set maxinal number of keypoints to find
        point_number = 1000 if config['max_num_keypoints'] is None else config['max_num_keypoints']

        savemat(matlab_config_path, {
            'file_list': file_list,
            'collection_names': collection_names,
            'set_names': set_names,
            'image_names': image_names,
            'dir_output': dir_output,
            'dir_data': dir_data,
            'point_number': point_number
        })
        subprocess.check_call(['matlab', '-nosplash', '-r',
        "point_extractor('vlfeat-0.9.21', '{}');quit".format(matlab_config_path)])

    except Exception as e:
        print('TCovDet: Keypoint feature map creation failed.')
        print(e)
        raise e

    # Load each created .mat file, extract the keypoints (Column 2 and 5),
    # create list of cv2.KeyPoint.
    # Then load the image, scale it, draw the keypoints in it and save everything
    for i in tqdm(range(len(file_list))):
        file = file_list[i]
        mat_path = os.path.join(config['tmp_dir_tcovdet'], 'feature_points', collection_names[i], set_names[i], image_names[i] + '.mat')
        kpts_numpy = loadmat(mat_path)['feature'][:, [2, 5]]        # numpy array

        if len(kpts_numpy):
            kpts_cv2 = [cv2.KeyPoint(x[0], x[1], 1.0) for x in kpts_numpy]   # list of cv2.KeyPoint

            img = cv2.imread(file, 0)
            if (img.shape[0] * img.shape[1]) > (1024 * 768):
                ratio = (1024 * 768 / float(img.shape[0] * img.shape[1]))**(0.5)
                img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation = cv2.INTER_CUBIC)

            img_kp = cv2.drawKeypoints(img, kpts_cv2, None)

            # Save everything.
            io_utils.save_detector_output(file, config['detector_name'], config,
                kpts_cv2, img_kp, None)
        else:
            print('Warning: Did not find any keypoints!')

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
    else:
        raise NotImplemented


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
