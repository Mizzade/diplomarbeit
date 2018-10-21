import torch
import tfeat_model
import os
import numpy as np
import cv2
from typing import List
import math


def load_tfeat(models_path: str='models', net_name: str='tfeat-liberty') -> tfeat_model.TNet:
    """Initialize tfeat and load the trained weights.

    Keyword Arguments:
        models_path {str} -- Path to the pretrained models. (default: {'models'})
        net_name {str} -- Name of the pretrained model. (default: {'tfeat-liberty'})

    Returns:
        tfeat_model.TNet -- The initialized neural network.
    """

    #init tfeat and load the trained weights
    tfeat = tfeat_model.TNet()
    tfeat.load_state_dict(torch.load(os.path.join(models_path,net_name+".params")))
    tfeat.cuda()
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
