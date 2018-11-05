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

def smart_scale(image: np.array, size: int) -> np.array:
    """Set the max size for the larger dimension of an image and scale the
    image accordingly. If `size` and the dimension of the image already
    correspond, return a copy of the image.

    Arguments:
        image {np.array} -- The image to be rescaled.
        size {int} -- Maximal size of the larger dimension of the image

    Returns:
        np.array -- Resized image, such that it's larger dimension is equal to
        `size`.
    """
    height, width = image.shape[:2]     # dimensions of image
    max_dim = np.max(image.shape[:2])   # max dimension of image
    interpolation = cv2.INTER_AREA      # Select interpolation algorithm
    scaling = size / max_dim            # Get scaling factor

    # If the largest iamge dimension already corresponds to the wanted size,
    # just return a copy of the image.
    if max_dim == size:
        return image.copy()

    if max_dim < size:
        interpolation = cv2.INTER_LINEAR # for upscaling

    return cv2.resize(image, None, fx=scaling, fy=scaling,
        interpolation=interpolation)

def compute_bundle(
    model: cv2.xfeatures2d_SIFT,
    image_list: List[str],
    size: int=None) -> List[Tuple[List[cv2.KeyPoint], np.array]]:
    """Computes keypoints, descriptors and images with keypoints drawn into it
    for a list of images. Returns a list of tuples. Each tuple contains
    the keypoints, the descriptors and the corresponding image with keypoints
    for each input image.

    Arguments:
        model {cv2.xfeatures2d_SIFT} -- The sift keypoint detector and descriptor.
        image_list {List[np.array]} -- A list of image paths
        size {None} -- Maximal dimension of image. Default: None.

    Returns:
        List[Tuple[List[cv2.KeyPoint], np.array]] -- List of 3-tuples containing
        the keypoints, descriptors and the image containing the keypoints.
    """

    output = []

    for image in image_list:
        output.append(compute(model, image, size))

    return output

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
        Tuple[List[cv2.KeyPoint], np.array, np.array] -- Returns tuple (keypoints, descriptors, image with keypoints).
    """

    img = cv2.imread(image, 0)
    img = smart_scale(img, size) if size is not None else img
    kp, desc = model.detectAndCompute(img, None)
    img_kp = cv2.drawKeypoints(img, kp, None)
    return (kp, desc, img_kp)

def save_output(
    file_list: List[str],
    output: List[Tuple[List[cv2.KeyPoint], np.array, np.array]],
    output_dir: str,
    detector_name,
    descriptor_name,
    project_name) -> None:
    """Save the output of this model inside the `output_dir`

    Arguments:
        file_list {List[str]} -- List of all file paths.
        output {List[Tuple[List[cv2.KeyPoint], np.array, np.array]]} -- The output of this model. In this case a triple of list of keypoints, descriptors, image with keypoints.
        output_dir {str} -- Path to the output directory
        detector_name {str} -- Name of the used detector
        descriptor_name {str} -- Name of the used descriptor
        project_name {str} -- Name of project.
    """

    for file_path, (kpts, desc, img) in zip (file_list, output):
        set_name, file_name, extension = io_utils.get_setName_fileName_extension(file_path)
        dir_path = os.path.join(output_dir, set_name)

        kp_path = io_utils.build_output_name(
            dir_path,
            file_name,
            detector_name=detector_name,
            prefix=os.path.join('keypoints',
                                'kpts_{}_'.format(project_name)))

        desc_path = io_utils.build_output_name(
            dir_path,
            file_name,
            descriptor_name=descriptor_name,
            prefix=os.path.join('descriptors',
                                'desc_{}_'.format(project_name)))

        kp_img_path = io_utils.build_output_name(
            dir_path,
            file_name,
            detector_name=detector_name,
            file_type='png',
            prefix=os.path.join('keypoint_images',
                                'kpts_{}_'.format(project_name)))

        io_utils.save_keypoints_list(kpts, kp_path, img.shape)
        io_utils.save_descriptors(desc, desc_path)
        io_utils.save_keypoints_image(img, kp_img_path)

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

    output_dir = argv[0]
    file_list = json.loads(argv[1])
    model = load_sift()
    size = None

    output = compute_bundle(model, file_list, size)
    save_output(file_list, output, output_dir, 'SIFT', 'SIFT', 'sift')

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
