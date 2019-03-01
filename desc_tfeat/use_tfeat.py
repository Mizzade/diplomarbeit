import torch
import tfeat_model
import os
import numpy as np
import cv2
from typing import List, Tuple, Any, Dict
import math
import io_utils
import sys
import json
from tqdm import tqdm
import pickle


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
        patch = img[i*patch_size:(i+1)*patch_size, :] # 65x65 Patch
        patch = io_utils.smart_scale(patch, 32, prevent_upscaling=True) # 32x32
        patches.append(patch)

    # TODO: Muss hier das Patch noch rektifiziert werden?
    desc = compute_descriptors(model, patches, use_gpu=False)

    return desc

def compute(image_file_path:str, config:Dict, model:Any) -> np.array:
    """Computes descriptors from keypoints saved in a file."""
    # Load image and scale appropiately. Image is later used to create patch,
    # which in turn is used to create the descriptor.
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
        max_size=config['max_size'])

    if not os.path.isfile(keypoints_file_path):
        print('Could not find keypoints in path: {}\n.Skip'.format(keypoints_file_path))
        return None

    # Load keypoints from csv file as numpy array.
    kpts_numpy = io_utils.get_keypoints_from_csv(keypoints_file_path)

    # Convert numpy array to List of cv2.KeyPoint list
    kpts_cv2 = io_utils.numpy_to_cv2_kp(kpts_numpy)

    # Create image patches for each keypoint
    patches = rectify_patches(img, kpts_cv2, 32, 3)

    #Compute descriptors
    desc = compute_descriptors(model, patches, use_gpu=False)

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
    model = load_tfeat()

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

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
