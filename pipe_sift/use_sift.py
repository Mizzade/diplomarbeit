from typing import List, Tuple
import cv2
import numpy as np
import sys
import json
import pickle
import copyreg
import os
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

def main(argv: List[str]) -> None:
    """Runs the SIFT model and saves the results.

    Arguments:
        argv {List[str]} -- List of parameters. The first paramters must be the
        path to the output directory where the result of this model will be
        saved. The second argument is a JSON-string, containing the list of all
        files that the model should work with.
    """

    assert isinstance(argv[0], str)
    assert isinstance(argv[1], str)
    assert isinstance(json.loads(argv[1]), list)

    project_name = 'sift'
    detector_name = 'SIFT'
    descriptor_name = 'SIFT'

    output_dir = argv[0]
    file_list = json.loads(argv[1])
    model = load_sift()
    size = 800

    for file in file_list:
        io_utils.save_output(file, compute(model, file, size), output_dir,
            detector_name, descriptor_name, project_name)

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
