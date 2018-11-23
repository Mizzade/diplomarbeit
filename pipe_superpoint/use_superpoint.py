from superpoint_frontend import SuperPointFrontend
import numpy as np
import torch
import cv2
from typing import List, Tuple
import os
import io_utils
import sys
import json
from tqdm import tqdm


def load_superpoint(\
    weights_path: str='models/superpoint_v1.pth',  # Path to pretrained weights file. \
    nms_dist: int=4,            # Non Maximum Suppression (NMS) distance. \
    conf_thresh: float=0.015,   # Detector confidende threshold. \
    nn_thresh: float=0.7,       # Descriptor matching threshold. \
    cuda: bool=False,            # Use cuda to speed up computation. \
    width: int=120,             # Input image height. \
    height: int=160             # Input image width. \
    ) -> SuperPointFrontend:
    return SuperPointFrontend(weights_path=weights_path,
                            nms_dist=nms_dist,
                            conf_thresh=conf_thresh,
                            nn_thresh=nn_thresh,
                            cuda=cuda,
                            width=width,
                            height=height)

def resize_image(image: np.array, width: int, height: int) -> np.array:
    """Resizes an image to given width and height. Returns a copy of input image with new dimensions.
    Automatically uses cv2.INTER_LINEAR for upscaling and cv2.INTER_AREA for downscaling.
    Returns a copy of same size, if shape of image and (height, width) do not
    differ.

    Arguments:
        image {np.array} -- An image loaded in openCV.
        width {int} -- New width of the input image.
        height {int} -- New height of the input image.

    Returns:
        resized_image {np.array} -- Copy of the input image with new dimensions.
    """

    image_shape = image.shape
    if height > image_shape[0] or width > image_shape[1]:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA

    if height == image_shape[0] and width == image_shape[1]:
        return image.copy()

    return cv2.resize(image, tuple((width, height)), interpolation=interpolation)


def convert_image_to_float32(image: np.array) -> np.array:
    """Convert a given image to be of dytpe `float32`.

    Arguments:
        image {np.array} -- An image with arbitrary dtype.

    Returns:
        converted_image {np.array} -- Copy of input image with dtype `float32`.
    """

    return image.astype('float32')

def normalize_image(image: np.array) -> np.array:
    """Normalizes the values of an input image.

    Arguments:
        image {np.array} -- An image of arbitrary type.

    Returns:
        normalized_image {np.array} -- Normalized copy of the input image.
    """

    return (image / 255.)


def detectAndCompute(model: SuperPointFrontend, image: np.array):
    img = resize_image(image, model.width, model.height)
    img = convert_image_to_float32(img)
    img = normalize_image(img)

    # pts: [3 x N] = [x, y, heatmap[x, y]]^T
    # desc = [256 x N]
    # heatmpa = [width x height]
    pts, desc, heatmap = model.run(img)

    return pts.T, desc.T, heatmap

def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - Lx3 numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]
    """
    # Check if descriptor dimensions match
    assert desc1.shape[1] == desc2.shape[1]

    # Return zero matches, if one image does not have a keypoint and
    # therefore no descriptors.
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
      return np.zeros((0, 3))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')

    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1, desc2.T)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))

    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[0])[keep]
    m_idx2 = idx
    # Populate the final Nx3 match data structure.
    matches = np.zeros((int(keep.sum()), 3))
    matches[:, 0] = m_idx1
    matches[:, 1] = m_idx2
    matches[:, 2] = scores
    return matches

def matches2DMatch(matches: np.array) -> List[cv2.DMatch]:
    """Transforms `nn_match_two_way`'s matches to a list of openCV's DMatch objects.

    Arguments:
        matches {np.array} -- Output of `nn_match_two_way` array of form  Nx3.

    Returns:
        List[cv2.DMatch] -- List of N openCV's DMatch elements.
    """

    return [cv2.DMatch(int(x[0]), int(x[1]), int(x[2])) for x in matches]

def kps2KeyPoints(kps: np.array) -> List[cv2.KeyPoint]:
    """Transforms SuperPoint's keypoints to openCV's list of Keypoints.

    Arguments:
        kps {np.array} -- Keypoints computed by SuperPoint. (Nx3)

    Returns:
        List[cv2.KeyPoint] -- List of N openCV's KeyPoint objects.
    """

    return [cv2.KeyPoint(x[0], x[1], x[2]) for x in kps]

def scale_kps(model: SuperPointFrontend, image: np.array, kps: np.array) -> np.array:
    """Scales keypoints `kps` up to match the image's dimensions.

    Arguments:
        model {SuperPointFrontend} -- SuperPoint instance.
        image {np.array} -- Original image with original size.
        kps {np.array} -- Keypoints computed by SuperPoint (Nx3).

    Returns:
        scaled_kps {np.array} -- The scaled keypoints to be used on the original image. (Nx3)
    """

    w, h = image.shape
    _fx = w / model.width
    _fy = h / model.height

    scaling = np.array([_fx, _fy, 1])
    return scaling * kps

def compute(
    model: SuperPointFrontend,
    image: str,
    size: int=None) -> Tuple[List[cv2.KeyPoint], np.array, np.array, np.array]:
    """Computes the keypoints and descriptors for a given input image.
    Draws keypoints into the image.
    Returns keypoints, descriptors and image with keypoints.

    Arguments:
        model {superpoint_frontend.SuperPointFrontend} -- The sift keypoint detector and descriptor.
        image {np.array} -- Path to the image.
        size {None} -- Maximal dimension of image. Default: None.

    Returns:
        Tuple[List[cv2.KeyPoint], np.array, np.array, None] -- Returns tuple (keypoints, descriptors, image with keypoints, image of heatmap).
    """

    img = cv2.imread(image, 0)
    img = io_utils.smart_scale(img, size, prevent_upscaling=True) if size is not None else img

    # Adjust values on the fly.
    model.height = img.shape[0]
    model.width = img.shape[1]

    _kp, desc, heatmap = detectAndCompute(model, img)
    kp = kps2KeyPoints(_kp)
    img_kp = cv2.drawKeypoints(img, kp, None)
    return (kp, desc, img_kp, heatmap)

def main(argv: List[str]) -> None:
    """Runs the SuperPoint model and saves the results.

    Arguments:
        argv {List[str]} -- List of parameters. The first paramters must be the
        path to the output directory where the result of this model will be
        saved. The second argument is a JSON-string, containing the list of all
        files that the model should work with.
    """

    assert isinstance(argv[0], str)
    assert isinstance(argv[1], str)
    assert isinstance(json.loads(argv[1]), list)

    project_name = 'superpoint'
    detector_name = 'SuperPoint'
    descriptor_name = 'SuperPoint'

    output_dir = argv[0]
    file_list = json.loads(argv[1])
    model = load_superpoint()
    size = 800

    for file in tqdm(file_list):
        io_utils.save_output(file, compute(model, file, size),
            output_dir, detector_name, descriptor_name, project_name)

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
