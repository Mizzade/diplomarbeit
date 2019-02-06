from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import sys
import os
from tqdm import tqdm
import io_utils
import pickle
import subprocess
from scipy.io import savemat

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

    print('Start trying')
    # try:
    #     # Create feature map for each image in 'covariant_points' folder
    #     subprocess.check_call(['python', 'patch_network_point_test.py',
    #     '--save_feature', 'covariant_points',
    #     '--output_dir', config['tmp_dir_tcovdet'],
    #     '--file_list', ' '.join(file_list),
    #     ])

    # except Exception as e:
    #     print('Something went terrible wrong.')
    #     print(e)

    try:
        # Create .mat files containing 'keypoints'.
        #dataset_name='webcam'
        #conv_feature_name='covariant_point_tilde'
        #feature_name='feature_point_tilde'
        #network_name='mexico_tilde_p24_Mexico_train_point_translation_iter_20'
        #stats_name='mexico_tilde_p24_Mexico_train_point'

        # point_number='1000'
        #$matlab -r "point_extractor('$dataset_name','$conv_feature_name','$feature_name',$point_number);";

        collection_names = []
        set_names = []
        image_bases = []
        for file_path in file_list:
            collection_name, set_name, file_base, extension = io_utils.get_path_components(file_path)
            collection_names.append(collection_name)
            set_names.append(set_name)
            image_bases.append(file_base)

        # The path to this file
        matlab_config_path = os.path.join(config['tmp_dir_tcovdet'], 'filelist.mat')
        # Where to save the keypoints
        dir_output = os.path.join(config['tmp_dir_tcovdet'], 'feature_points')
        io_utils.create_dir(dir_output) # Create on the fly if not existent

        # Where the .mat files of the covariant step lie
        dir_data = os.path.join(config['tmp_dir_tcovdet'], 'covariant_points')

        savemat(matlab_config_path, {
            'file_list': file_list,
            'collection_names': collection_names,
            'set_names': set_names,
            'image_bases': image_base,
            'dir_output': dir_output,
            'dir_data': dir_data,
            'point_number': 1000
        })
        subprocess.check_call(['matlab', '-nosplash', '-r',
        "point_extractor('vlfeat-0.9.21', '{}')".format(matlab_config_path)])

        # subprocess.check_call(['matlab', '-nosplash', '-r',
        # "test_program('vlfeat-0.9.21', 'matconvnet-1.0-beta25', 'HPatches_ST_LM_128d.mat', '.', '{}', '{}');quit".format(path_to_patches, path_to_desc)])
    except Exception as e:
        print('oopsie.')
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
