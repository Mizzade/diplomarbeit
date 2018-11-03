from typing import List
import cv2
import sys
import os
import numpy as np

def build_output_name(dir_path: str, image_name: str, detector_name: str='', prefix='', descriptor_name: str='', file_type: str='csv') -> str:

    # TODO assert that at least detector_name or descriptor_name is set
    if not detector_name and not descriptor_name:
        sys.exit('Error: `detector_name` or `descriptor_name` must be set')

    _list = [x for x in [prefix, image_name, detector_name, descriptor_name] if x]

    return os.path.join(dir_path, '_'.join(_list) + '.{}'.format(file_type))

def _save_keypoints_list(kpts: List[cv2.KeyPoint], file_name: str) -> None:
    _a = np.array([[*x.pt, x.size, x.angle, x.response, x.octave, x.class_id] for x in kpts])
    np.savetxt(file_name,
        _a,
        delimiter=',',
        header='x, y, size, angle, response, octave, class_id\n{}, {}'.format(_a.shape[0], _a.shape[1]))

def save_keypoints_list(kpts: List[cv2.KeyPoint], file_name: str) -> None:
    _dirname = os.path.dirname(file_name)
    if not os.path.exists(_dirname):
        os.makedirs(_dirname, exist_ok=True)
    _save_keypoints_list(kpts, file_name)

def _save_descriptors(desc: np.array, file_name: str) -> None:
    np.savetxt(file_name,
        desc,
        delimiter=',',
        header='{}, {}'.format(desc.shape[0], desc.shape[1]))

def save_descriptors(desc: np.array, file_name: str) -> None:
    _dirname = os.path.dirname(file_name)
    if not os.path.exists(_dirname):
        os.makedirs(_dirname, exist_ok=True)
    _save_descriptors(desc, file_name)

def _save_keypoints_image(image: np.array, file_name: str) -> None:
    cv2.imwrite(file_name, image)

def save_keypoints_image(image: np.array, file_name: str) -> None:
    _dirname = os.path.dirname(file_name)
    if not os.path.exists(_dirname):
        os.makedirs(_dirname, exist_ok=True)
    _save_keypoints_image(image, file_name)

def get_setName_fileName_extension(file_path: str) -> (str, str, str):
    # ...foo/bar/<set_name>/<file_name>.<extensions>
    _base_path, extension = os.path.splitext(file_path)
    set_name, file_name = _base_path.split(os.sep)[-2:]

    return set_name, file_name, extension

