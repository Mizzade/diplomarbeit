from typing import List, Tuple, Any, Dict
import cv2
import numpy as np
import sys
import json
import os
import io_utils
import subprocess
import shutil
from tqdm import tqdm
import pickle

def create_patches(img: np.array, kpts: List[cv2.KeyPoint], N: int) -> List[np.array]:
    """Creats a csv file containing patches around the keypoints given in `kpts`.

    Arguments:
        img {np.array} -- The image to get the patches from.
        kpts {List[cv2.KeyPoint]} -- List of M keypoints.
        N {int} -- Size of square patches in pixel.

    Returns:
        List[np.array] -- List of Mx(NxN) patches.
    """
    patches = []
    for kp in kpts:
        patches.append(cv2.getRectSubPix(img, (N, N), kp.pt))

    return patches

def computeForPatchImages(image_file_path:str, config:Dict, model:Any) -> np.array:
    """Computes descriptors for images containing patches to be described."""

    # Load patch image
    img = cv2.imread(image_file_path, 0)

    # Assuming the patches are ordered vertically, and all patches are squares
    # of size MxM, find number of patches in image and compute the descriptor
    # for each patch.
    patch_size = img.shape[1]
    num_patches = np.int(img.shape[0] / patch_size)

    patches = []
    for i in range(num_patches):
        patch = img[i*patch_size:(i+1)*patch_size, :]
        patch = io_utils.smart_scale(patch, 42)
        patches.append(patch)

    # Save patches in tmp dir
    path_to_desc = os.path.join(config['tmp_dir_doap'], 'descriptors.csv')
    path_to_patches = os.path.join(config['tmp_dir_doap'], 'patches.csv')
    io_utils.save_patches_list(patches, path_to_patches)

    # Compute descritpors in matlab. Save result in tmp_dir
    # TODO: file paths for vlfeat, matconvnet and the model must be parameters
    subprocess.check_call(['matlab', '-nosplash', '-r',
    "use_doap_with_file('vlfeat-0.9.21', 'matconvnet-1.0-beta25', 'HPatches_ST_LM_128d.mat', '.', '{}', '{}');quit".format(path_to_patches, path_to_desc)])

    # Load matlab results and return.
    desc = np.loadtxt(path_to_desc, delimiter=',')

    return desc


def compute(image_file_path:str, config:Dict, model:Any) -> np.array:
    """Computes descriptors from keypoints saved in a file."""

    img = cv2.imread(image_file_path, 0)
    img = io_utils.smart_scale(img, config['max_size'], prevent_upscaling=True) if config['max_size'] is not None else img

     # Infer the path to the corresponding csv file for the keypoints.
    collection_name, set_name, image_name, _ = io_utils.get_path_components(image_file_path)

    # find path to keypoints file
    keypoints_file_path = io_utils.build_output_path(
        config['output_dir'],
        collection_name,
        set_name,
        'keypoints',
        config['detector_name'],
        image_name,
        max_size=config['max_size'],
        max_num_keypoints=config['max_num_keypoints'])

    if not os.path.isfile(keypoints_file_path):
        print('Could not find keypoints in path: {}\n.Skip'.format(keypoints_file_path))
        return None

    # Load keypoints from csv file as numpy array.
    kpts_numpy = io_utils.get_keypoints_from_csv(keypoints_file_path)

    # Convert numpy array to List of cv2.KeyPoint list
    kpts_cv2 = io_utils.numpy_to_cv2_kp(kpts_numpy)

    # Create iamge patches for each keypoint
    patches = create_patches(img, kpts_cv2, 42)

    # Save patches in tmp dir
    path_to_desc = os.path.join(config['tmp_dir_doap'], 'descriptors.csv')
    path_to_patches = os.path.join(config['tmp_dir_doap'], 'patches.csv')
    io_utils.save_patches_list(patches, path_to_patches)

    # Compute descritpors in matlab. Save result in tmp_dir
    # TODO: file paths for vlfeat, matconvnet and the model must be parameters
    subprocess.check_call(['matlab', '-nosplash', '-r',
    "use_doap_with_file('vlfeat-0.9.21', 'matconvnet-1.0-beta25', 'HPatches_ST_LM_128d.mat', '.', '{}', '{}');quit".format(path_to_patches, path_to_desc)])

    # Load matlab results and return.
    desc = np.loadtxt(path_to_desc, delimiter=',')

    return desc

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

    config, file_list = config_file
    model = None

    if config['task'] == 'descriptors':
        for file in tqdm(file_list):
            descriptors = compute(file, config, model)
            if descriptors is not None:
                io_utils.save_descriptor_output(file, config, descriptors)

    elif config['task'] == 'patches':
        for file in tqdm(file_list):
            descriptors = computeForPatchImages(file, config, model)
            if descriptors is not None:
                io_utils.save_descriptor_output(file, config, descriptors)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
