from typing import List, Tuple, Dict, Any, Callable
import cv2
import numpy as np
import sys
import json
import os
from tqdm import tqdm
import io_utils
import pickle

def load_sift(config:Dict) -> cv2.xfeatures2d_SIFT:
    """Creates an SIFT instance.

    Arguments:
        config {dict} -- config object. See config_run_detectors.py for more
            information.

    Returns:
        cv2.xfeatures2d_SIFT -- The SIFT keypoint detector and desriptor.
    """

    if config['task'] == 'keypoints':
        nfeatures = 0 if config['max_num_keypoints'] is None else config['max_num_keypoints']
    else:
        nfeatures = 0


    return cv2.xfeatures2d.SIFT_create(nfeatures)

def computeForPatchImages(image_file_path:str, config:Dict, model:Any) -> np.array:
    """Computes descriptors for images containing patches to be described."""
    # Load patch image

    img = cv2.imread(image_file_path, 0)

    # Assuming the patches are ordered vertically, and all patches are squares
    # of size MxM, find number of patches in image and compute the descriptor
    # for each patch.
    patch_size = img.shape[1]
    num_patches = np.int(img.shape[0] / patch_size)
    patch_center = np.int(0.5 * (patch_size - 1))

    # keypoint at patch center with size (diameter) of patch
    kp = cv2.KeyPoint(patch_center, patch_center, patch_size)

    desc = []

    for i in range(num_patches):
        patch = img[i*patch_size:(i+1)*patch_size, :]
        _, d = model.compute(patch, [kp])
        desc.append(d.flatten())

    desc = np.vstack(desc)
    print('desc: ', desc)

    return desc


def compute(image_file_path:str, config:Dict, model:Any) -> np.array:
    """Computes descriptors from keypoints saved in a file.
    """

    # Load image
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

    # Compute descriptors
    _, desc = model.compute(img, kpts_cv2)

    return desc

def detect(image_path:str, config:Dict, detector:Any) -> Tuple[List, np.array, None]:
    """Detects keypoints for a given input image.
    Draws keypoints into the image.
    Returns keypoints, heatmap and image with keypoints.
    """
    img = cv2.imread(image_path, 0)
    img = io_utils.smart_scale(img, config['max_size'], prevent_upscaling=config['prevent_upscaling']) if config['max_size'] is not None else img

    # Get keypoints
    kpts = detector.detect(img, None)

    # Sort by response, take best n <= max_num_keypoints
    kpts.sort(key=lambda x: x.response, reverse=True)

    img_kp = io_utils.draw_keypoints(img, kpts, config)
    return (kpts, img_kp, None)

def main(argv: Tuple[str]) -> None:
    """Runs the TILDE model and saves the results.

    Arguments:
        argv {Tuple[str]} -- List of one parameter. There should be exactly
            one parameter - the path to the config file inside the tmp dir.
            This config file will be used to get all other information and
    """
    if len(argv) <= 0:
        raise RuntimeError("Missing argument <path_to_config_file>. Abort")

    with open(argv[0], 'rb') as src:
        config_file = pickle.load(src, encoding='utf-8')

    config, file_list = config_file
    model = load_sift(config)

    if config['task'] == 'keypoints':
        for file in tqdm(file_list):
            keypoints, keypoints_image, heatmap_image = detect(file, config, model)
            io_utils.save_detector_output(
                file,
                config['detector_name'],
                config,
                keypoints,
                keypoints_image,
                heatmap_image)

    elif config['task'] == 'descriptors':
        for file in tqdm(file_list):
            descriptors = compute(file, config, model)
            if descriptors is not None:
                io_utils.save_descriptor_output(file, config, descriptors)

    elif config['task'] == 'patches':
        for file in tqdm(file_list):
            descriptors = computeForPatchImages(file, config, model)
            if descriptors is not None:
                io_utils.save_descriptor_output(file, config, descriptors)


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
