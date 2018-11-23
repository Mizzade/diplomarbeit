from typing import List, Tuple, Any
import cv2
import numpy as np
import sys
import json
import os
import io_utils
import subprocess
import shutil
from tqdm import tqdm

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



def compute(
    detector: Any,
    model: Any,
    image: str,
    size: int=None) -> Tuple[List[cv2.KeyPoint], np.array, np.array, np.array]:
    """Computes the keypoints and descriptors for a given input image.
    Draws keypoints into the image.
    Returns keypoints, descriptors and image with keypoints.

    Arguments:
        detector {Any} -- A keypoint detector.
        model {None} -- A model to load in python.
        image {np.array} -- Path to the image.
        size {None} -- Maximal dimension of image. Default: None.

    Returns:
        Tuple[List[cv2.KeyPoint], np.array, np.array, None] -- Returns tuple (keypoints, descriptors, image with keypoints, image of heatmap).
    """
    """Wrapper function for DOAP.
    1) Create a temporay folder `tmp` to save intermediate output.
    2a) Load and smart scale the image.
    2b) Find keypoints inside that image.
    2c) For each keypoint get a 42x42 patch around that keypoint and save that
        in the corresponding directory in the `tmp` dir.
    3) Call now use_doap_with_file.m for the `pathes` file in the `tmp`
        directory and compute the descriptors.
    4) Draw list of cv2.KeyPoints into image.
    5) Return KeyPoint list, descriptors and image with KeyPoints.
    """
    base_path, _ =  os.path.splitext(os.path.abspath(__file__))
    base_path = os.sep.join(base_path.split(os.sep)[:-1])

    # 1)
    path_tmp = os.path.join(base_path, 'tmp')
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp, exist_ok=True)

    # 2a)
    img = cv2.imread(image, 0)
    img = io_utils.smart_scale(img, size, prevent_upscaling=True) if size is not None else img

    # 2b)
    kp, _ = detector.detectAndCompute(img, None)

    # 2c)
    patches = create_patches(img, kp, 42) # list of patches
    path_to_patches = os.path.join(path_tmp, 'patches.csv')
    path_to_desc = os.path.join(path_tmp, 'descriptors.csv')
    io_utils.save_patches_list(patches, path_to_patches)

    # 3)
    # TODO: file paths for vlfeat, matconvnet and the model must be paramters
    subprocess.check_call(['matlab', '-nosplash', '-r',
    "use_doap_with_file('vlfeat-0.9.21', 'matconvnet-1.0-beta25', 'HPatches_ST_LM_128d.mat', '.', '{}', '{}');quit".format(path_to_patches, path_to_desc)])

    desc = np.loadtxt(path_to_desc, delimiter=',')

    # 4)
    img_kp = cv2.drawKeypoints(img, kp, None)

    # 5)
    return (kp, desc, img_kp, None)


def main(argv: Tuple[str, str,str]) -> None:
    """Runs the DOAP model and saves the results.

    Arguments:
        argv {Tuple[str, str, str]} -- List of parameters. Expects exactly three
            parameters. The first one contains json-fied network information,
            the second contains the json-fied config object and the third is
            the json-fied file list with all files to be processed.
    """

    project_name = 'doap'
    detector_name = 'SIFT'
    descriptor_name = 'DOAP'

    network = json.loads(argv[0])
    config = json.loads(argv[1])
    file_list = json.loads(argv[2])
    detector = cv2.xfeatures2d.SIFT_create()
    model = None

    for file in tqdm(file_list):
        io_utils.save_output(
            file,
            compute(detector, None, file, config['size']),
            config['output_dir'],
            detector_name,
            descriptor_name,
            project_name)

    # Clean up
    base_path, _ =  os.path.splitext(os.path.abspath(__file__))
    base_path = os.sep.join(base_path.split(os.sep)[:-1])
    path_tmp = os.path.join(base_path, 'tmp')
    shutil.rmtree(path_tmp, ignore_errors=True)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
