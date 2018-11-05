from typing import List
import cv2
import sys
import os
import numpy as np

def build_output_name(dir_path: str, image_name: str, detector_name: str='',
    prefix='', descriptor_name: str='', file_type: str='csv') -> str:
    # TODO assert that at least detector_name or descriptor_name is set
    if not detector_name and not descriptor_name:
        sys.exit('Error: `detector_name` or `descriptor_name` must be set')

    _list = [x for x in [prefix, image_name, detector_name, descriptor_name] if x]

    return os.path.join(dir_path, '_'.join(_list) + '.{}'.format(file_type))

def save_keypoints_list(kpts: List[cv2.KeyPoint], file_name: str, image_shape: np.array) -> None:
    _dirname = os.path.dirname(file_name)
    if not os.path.exists(_dirname):
        os.makedirs(_dirname, exist_ok=True)

    _a = np.array([[*x.pt, x.size, x.angle, x.response, x.octave, x.class_id] for x in kpts])
    np.savetxt(file_name,_a, delimiter=',',
        header='height, width, number of rows, number of columns\n{}, {}, {}, {}\nx, y, size, angle, response, octave, class_id' \
            .format(image_shape[0], image_shape[1], _a.shape[0], _a.shape[1]))

def save_descriptors(desc: np.array, file_name: str) -> None:
    _dirname = os.path.dirname(file_name)
    if not os.path.exists(_dirname):
        os.makedirs(_dirname, exist_ok=True)
    np.savetxt(file_name, desc, delimiter=',',
        header='{}, {}'.format(desc.shape[0], desc.shape[1]))

def save_keypoints_image(image: np.array, file_name: str) -> None:
    _dirname = os.path.dirname(file_name)
    if not os.path.exists(_dirname):
        os.makedirs(_dirname, exist_ok=True)
    cv2.imwrite(file_name, image)

def get_setName_fileName_extension(file_path: str) -> (str, str, str):
    # ...foo/bar/<set_name>/<file_name>.<extensions>
    _base_path, extension = os.path.splitext(file_path)
    set_name, file_name = _base_path.split(os.sep)[-2:]

    return set_name, file_name, extension

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
