from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import sys
import json
import os
from tqdm import tqdm
import io_utils
import pickle

def load_sift() -> cv2.xfeatures2d_SIFT:
    """Creates an SIFT instance.

    Returns:
        cv2.xfeatures2d_SIFT -- The SIFT keypoint detector and desriptor.
    """

    return cv2.xfeatures2d.SIFT_create()

# def compute(
#     model: cv2.xfeatures2d_SIFT,
#     image: str,
#     size: int=None) -> Tuple[List[cv2.KeyPoint], np.array, np.array]:
#     """Computes the keypoints and descriptors for a given input image.
#     Draws keypoints into the image.
#     Returns keypoints, descriptors and image with keypoints.

#     Arguments:
#         model {cv2.xfeatures2d_SIFT} -- The sift keypoint detector and descriptor.
#         image {np.array} -- Path to the image.
#         size {None} -- Maximal dimension of image. Default: None.

#     Returns:
#         Tuple[List[cv2.KeyPoint], np.array, np.array, None] -- Returns tuple (keypoints, descriptors, image with keypoints, image of heatmap).
#     """

#     img = cv2.imread(image, 0)
#     img = io_utils.smart_scale(img, size, prevent_upscaling=True) if size is not None else img
#     kp, desc = model.detectAndCompute(img, None)
#     img_kp = cv2.drawKeypoints(img, kp, None)
#     return (kp, desc, img_kp, None)

def compute(image_file_path:str, config:Dict, model:Any) -> np.array:
    """Computes descriptors from keypoints saved in a file.
    """

    # Load image
    img = cv2.imread(image_file_path)

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
        config['max_size']
    )
    # Load keypoints from csv file as numpy array.
    kpts_numpy = io_utils.get_keypoints_from_csv(keypoints_file_path)


    # Convert numpy array to List of cv2.KeyPoint list
    kpts_cv2 = []
    for kp in kpts_numpy:
        x, y, _size, _angle, _response, _octave, _class_id = kp
        x, y, _size, _angle, _response, _octave, _class_id = \
            x, y, _size, _angle, _response, int(_octave), int(_class_id)

        kpts_cv2.append(cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id))

    # Compute descriptors
    _, desc = model.compute(img, kpts_cv2)

    return desc

def detect(image_path:str, config:Dict, detector:Any) -> None:
    """Detects keypoints for a given input image.
    Draws keypoints into the image.
    Returns keypoints, heatmap and image with keypoints.
    """
    img = cv2.imread(image_path, 0)
    img = io_utils.smart_scale(img, config['max_size'], prevent_upscaling=True) if config['max_size'] is not None else img
    kpts = detector.detect(img, None)
    img_kp = cv2.drawKeypoints(img, kpts, None)
    return (kpts, img_kp, None)

def main(argv: Tuple[str]) -> None:
    """Runs the TILDE model and saves the results.

    Arguments:
        argv {Tuple[str]} -- List of one parameters. There should be exactly
            one parameter - the path to the config file inside the tmp dir.
            This config file will be used to get all other information and
    """
    if len(argv) <= 0:
        raise RuntimeError("Missing argument <path_to_config_file>. Abort")

    with open(argv[0], 'rb') as src:
        config_file = pickle.load(src, encoding='utf-8')

    config, file_list = config_file
    model = load_sift()

    if config['task'] == 'keypoints':
        for file in tqdm(file_list):
            keypoints, keypoints_image, heatmap_image = detect(file, config, model)
            io_utils.save_detector_output(file, config['detector_name'], config, keypoints,
                keypoints_image, heatmap_image)

    elif config['task'] == 'descriptors':
        for file in tqdm(file_list):
            descriptors = compute(file, config, model)
            io_utils.save_descriptor_output(file, config, descriptors)


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
