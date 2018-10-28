from typing import List, Tuple
import cv2
import numpy as np

def load_sift() -> cv2.xfeatures2d_SIFT:
    """Creates an SIFT instance.

    Returns:
        cv2.xfeatures2d_SIFT -- The SIFT keypoint detector and desriptor.
    """

    return cv2.xfeatures2d.SIFT_create()

def compute_bundle(model: cv2.xfeatures2d_SIFT, image_list: List[str]) -> List[Tuple[List[cv2.KeyPoint], np.array]]:
    """Computes keypoints, descriptors and images with keypoints drawn into it
    for a list of images. Returns a list of tuples. Each tuple contains
    the keypoints, the descriptors and the corresponding image with keypoints
    for each input image.

    Arguments:
        model {cv2.xfeatures2d_SIFT} -- The sift keypoint detector and descriptor.
        image_list {List[np.array]} -- A list of image paths

    Returns:
        List[Tuple[List[cv2.KeyPoint], np.array]] -- List of 3-tuples containing
        the keypoints, descriptors and the image containing the keypoints.
    """

    output = []

    for image in image_list:
        output.append(compute(model, image))

    return output

def compute(model: cv2.xfeatures2d_SIFT, image: str) -> Tuple[List[cv2.KeyPoint], np.array, np.array]:
    """Computes the keypoints and descriptors for a given input image.
    Draws keypoints into the image.
    Returns keypoints, descriptors and image with keypoints.

    Arguments:
        model {cv2.xfeatures2d_SIFT} -- The sift keypoint detector and descriptor.
        image {np.array} -- Path to the image.

    Returns:
        Tuple[List[cv2.KeyPoint], np.array, np.array] -- Returns tuple (keypoints, descriptors, image with keypoints).
    """

    img = cv2.imread(image, 0)
    kp, desc = model.detectAndCompute(img, None)
    img_kp = cv2.drawKeypoints(img, kp, None)
    return (kp, desc, img_kp)
