import torch
import tfeat_model
import os
import numpy as np
import cv2
from typing import List, Tuple, Any
import math
import io_utils
import sys
import json
from tqdm import tqdm


def load_tfeat(models_path: str='models', net_name: str='tfeat-liberty', use_gpu=False) -> tfeat_model.TNet:
    """Initialize tfeat and load the trained weights.

    Keyword Arguments:
        models_path {str} -- Path to the pretrained models. (default: {'models'})
        net_name {str} -- Name of the pretrained model. (default: {'tfeat-liberty'})

    Returns:
        tfeat_model.TNet -- The initialized neural network.
    """

    #init tfeat and load the trained weights
    tfeat = tfeat_model.TNet()
    if use_gpu:
        tfeat.load_state_dict(
            torch.load(os.path.join(models_path,net_name+".params")))
        tfeat.cuda()
    else:
        tfeat.load_state_dict(
            torch.load(os.path.join(models_path,net_name+".params"), map_location='cpu'))
    tfeat.eval()
    return tfeat

def rectify_patches(img: np.array, kpts: List[cv2.KeyPoint], N: int, mag_factor: int) -> List[np.array]:
    """"Rectifies patches around openCV keypoints, and returns patches
    a list of of patches.

    Arguments:
        img {np.array} -- The image corresponding to the keypoints.
        kpts {List[cv2.KeyPoint]} -- List of keypoints found in the image using a detector.
        N {int} -- Size of the patches to extract.
        mag_factor {int} -- Magnification factor. Determines how many times the original keypoint scale is enlarged to generate a patch from a keypoint.

    Returns:
        List[np.array] -- List of extracted patches.
    """


    patches = []
    for kp in kpts:
        x,y = kp.pt
        s = kp.size
        a = kp.angle

        s = mag_factor * s / N
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.matrix([
            [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
            [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y]])

        patch = cv2.warpAffine(img, M, (N, N),
                             flags=cv2.WARP_INVERSE_MAP + \
                             cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)

        patches.append(patch)
    return patches

def compute_descriptors(model:tfeat_model.TNet, list_of_patches: List[np.array], use_gpu=True) -> np.array:
    """Transforms `list_of_patches` to **torch.tensor** and computes for each patch the corresponding descriptor.

    Arguments:
        model {tfeat.model.TNet} -- Tfeat neural network.
        list_of_patches {List[np.array]} -- List of N patches. Each patch is a 32x32pixels np.array with dtype=uint8.

    Keyword Arguments:
        use_gpu {bool} -- Use graphics card for computation. (default: {True})

    Returns:
        np.array -- An np.array of shape Nx128. Each i-th row is the descriptor of the corresponding i-th keypoint patch.
    """
    patches = torch.from_numpy(np.asarray(list_of_patches)).float()
    patches = torch.unsqueeze(patches,1)
    if use_gpu:
        patches = patches.cuda()
    descrs = model(patches)
    return descrs.detach().cpu().numpy()

def compute(
    detector: Any,
    model: tfeat_model.TNet,
    image: str,
    size: int=None) -> Tuple[List[cv2.KeyPoint], np.array, np.array, np.array]:
    """Computes the keypoints and descriptors for a given input image.
    Draws keypoints into the image.
    Returns keypoints, descriptors and image with keypoints.

    Arguments:
        detector {Any} -- A keypoint detector.
        model {tfeat_model.TNet} -- The keypoint detector and descriptor.
        image {np.array} -- Path to the image.
        size {None} -- Maximal dimension of image. Default: None.

    Returns:
        Tuple[List[cv2.KeyPoint], np.array, np.array, None] -- Returns tuple (keypoints, descriptors, image with keypoints, image of heatmap).
    """

    img = cv2.imread(image, 0)
    img = io_utils.smart_scale(img, size, prevent_upscaling=True) if size is not None else img
    kp, _ = detector.detectAndCompute(img, None)
    patches = rectify_patches(img, kp, 32, 3)
    desc = compute_descriptors(model, patches, use_gpu=False)

    img_kp = cv2.drawKeypoints(img, kp, None)
    return (kp, desc, img_kp, None)

def main(argv: Tuple[str, str,str]) -> None:
    """Runs the TFeat model and saves the results.

    Arguments:
        argv {Tuple[str, str, str]} -- List of parameters. Expects exactly three
            parameters. The first one contains json-fied network information,
            the second contains the json-fied config object and the third is
            the json-fied file list with all files to be processed.
    """

    project_name = 'tfeat'
    detector_name = 'SIFT'
    descriptor_name = 'TFeat'

    network = json.loads(argv[0])
    config = json.loads(argv[1])
    file_list = json.loads(argv[2])
    model = load_tfeat()
    detector = cv2.xfeatures2d.SIFT_create()

    for file in tqdm(file_list):
        io_utils.save_output(
            file,
            compute(detector, model, file, config['size']),
            config['output_dir'],
            detector_name,
            descriptor_name,
            project_name,
            config['size'])

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
