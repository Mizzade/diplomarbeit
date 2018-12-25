from typing import List, Tuple
import cv2
import numpy as np
import sys
import json
import os
from tqdm import tqdm
import io_utils

def load_sift() -> cv2.xfeatures2d_SIFT:
    """Creates an SIFT instance.

    Returns:
        cv2.xfeatures2d_SIFT -- The SIFT keypoint detector and desriptor.
    """

    return cv2.xfeatures2d.SIFT_create()

def compute(
    model: cv2.xfeatures2d_SIFT,
    image: str,
    size: int=None) -> Tuple[List[cv2.KeyPoint], np.array, np.array]:
    """Computes the keypoints and descriptors for a given input image.
    Draws keypoints into the image.
    Returns keypoints, descriptors and image with keypoints.

    Arguments:
        model {cv2.xfeatures2d_SIFT} -- The sift keypoint detector and descriptor.
        image {np.array} -- Path to the image.
        size {None} -- Maximal dimension of image. Default: None.

    Returns:
        Tuple[List[cv2.KeyPoint], np.array, np.array, None] -- Returns tuple (keypoints, descriptors, image with keypoints, image of heatmap).
    """

    img = cv2.imread(image, 0)
    img = io_utils.smart_scale(img, size, prevent_upscaling=True) if size is not None else img
    kp, desc = model.detectAndCompute(img, None)
    img_kp = cv2.drawKeypoints(img, kp, None)
    return (kp, desc, img_kp, None)

def main(argv: Tuple[str, str,str]) -> None:
    """Runs the SIFT model and saves the results.

    Arguments:
        argv {Tuple[str, str, str]} -- List of parameters. Expects exactly three
            parameters. The first one contains json-fied network information,
            the second contains the json-fied config object and the third is
            the json-fied file list with all files to be processed.
    """

    project_name = 'sift'
    detector_name = 'SIFT'
    descriptor_name = 'SIFT'

    network = json.loads(argv[0])
    config = json.loads(argv[1])
    file_list = json.loads(argv[2])
    model = load_sift()

    for file in tqdm(file_list):
        io_utils.save_output(
            file,
            compute(model, file, config['size']),
            config['output_dir'],
            detector_name,
            descriptor_name,
            project_name,
            config['size'])

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
