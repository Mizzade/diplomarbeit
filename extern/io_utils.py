from typing import List, Tuple, Any
import cv2
import sys
import os
import numpy as np
import shutil

def create_tmp_dir(path: str):
    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

def remove_tmp_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def build_output_name(dir_path: str, image_name: str, detector_name: str='',
    prefix='', descriptor_name: str='', size: int=None, file_type: str='csv') -> str:
    # TODO assert that at least detector_name or descriptor_name is set
    if not detector_name and not descriptor_name:
        sys.exit('Error: `detector_name` or `descriptor_name` must be set')


    _list = [x for x in [prefix, image_name, detector_name, descriptor_name] if x]

    if size is not None:
        _list.append(str(size))

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

def save_patches_list(patches: List[np.array], file_name:str) -> None:
    _dirname = os.path.dirname(file_name)
    if not os.path.exists(_dirname):
        os.makedirs(_dirname, exist_ok=True)

    _a = np.vstack(patches)
    np.savetxt(file_name, _a, delimiter=',')

def get_setName_fileName_extension(file_path: str) -> (str, str, str):
    # ...foo/<collection_name>/<set_name>/<file_name>.<extensions>
    _base_path, extension = os.path.splitext(file_path)
    collection_name, set_name, file_name = _base_path.split(os.sep)[-3:]

    return collection_name, set_name, file_name, extension

def smart_scale(image: np.array, size: int, prevent_upscaling: bool=False) -> np.array:
    """Set the max size for the larger dimension of an image and scale the
    image accordingly. If `size` and the dimension of the image already
    correspond, return the image
    If prevent_upscaling is set, return image if size is larger then the largest
    dimension of the image.

    Arguments:
        image {np.array} -- The image to be rescaled.
        size {int} -- Maximal size of the larger dimension of the image

    Returns:
        np.array -- Resized image, such that it's larger dimension is equal to
        `size`.
    """
    max_dim = np.max(image.shape[:2])   # max dimension of image
    interpolation = cv2.INTER_AREA      # Select interpolation algorithm
    scaling = size / max_dim            # Get scaling factor

    # If the largest iamge dimension already corresponds to the wanted size,
    # just return the image
    if max_dim == size:
        return image

    if max_dim < size:
        interpolation = cv2.INTER_LINEAR # for upscaling

    return cv2.resize(image, None, fx=scaling, fy=scaling,
        interpolation=interpolation)

def save_outputs(
    file_list: List[str],
    output_list: List[Tuple[List[cv2.KeyPoint], np.array, np.array, np.array]],
    output_dir: str,
    detector_name: str,
    descriptor_name: str,
    project_name: str,
    size: int=None) -> None:
    """Wrapper function for save_output. Saves the outputs of a model for each
    file given in the file_list to `output dir`.

    Arguments:
        file_list {List[str]} -- List of all file paths.
        output_list {List[Tuple[List[cv2.KeyPoint], np.array, np.array]]} -- List of output of this model for all files in `file_list`.
        Expects a 4-tuple, containing a list of of keypoints, descriptors,
        image with keypoints and a heatmap. If a model does not provide one of
        those four elements, use a list of None instead.
        output_dir {str} -- Path to the output directory
        detector_name {str} -- Name of the used detector
        descriptor_name {str} -- Name of the used descriptor
        project_name {str} -- Name of project.
        size {int} -- Smart scale size paramter used. If None, no size parameter
        was given when calling run_models. (Default: None)
    """
    for file_path, output in zip(file_path, output_list):
        save_output(file_path, output, output_dir, detector_name, descriptor_name, project_name)


def save_output(
    file_path: str,
    output: Tuple[List[cv2.KeyPoint], np.array, np.array, np.array],
    output_dir: str,
    detector_name: str,
    descriptor_name: str,
    project_name: str,
    size: int = None) -> None:
    """Save the output of a model inside `output_dir`

    Arguments:
        file_path {str} -- File path to file.
        output {Tuple[List[cv2.KeyPoint], np.array, np.array]} -- Output for this model and the file at file_path.
        Expects a 4-tuple, containing a list of of keypoints, descriptors,
        image with keypoints and a heatmap. If a model does not provide one of
        those four elements, use a list of None instead.
        output_dir {str} -- Path to the output directory
        detector_name {str} -- Name of the used detector
        descriptor_name {str} -- Name of the used descriptor
        project_name {str} -- Name of project.
        size {int} -- Smart scale size paramter used. If None, no size parameter
        was given when calling run_models. (Default: None
    """
    kpts, desc, img_kp, img_heatmap = output
    collection_name, set_name, file_name, _ = get_setName_fileName_extension(file_path)
    dir_path = os.path.join(output_dir, collection_name, set_name)

    # Save list of keypoints.
    if kpts is not None:
        kp_path = build_output_name(
            dir_path,
            file_name,
            size=size,
            detector_name=detector_name,
            prefix=os.path.join('keypoints',
                                'kpts_{}_'.format(project_name)))
        save_keypoints_list(kpts, kp_path, img_kp.shape)

    # Save descriptors
    if desc is not None:
        desc_path = build_output_name(
            dir_path,
            file_name,
            size=size,
            detector_name=detector_name,
            descriptor_name=descriptor_name,
            prefix=os.path.join('descriptors',
                                'desc_{}_'.format(project_name)))
        save_descriptors(desc, desc_path)

    # Save keypoint image
    if img_kp is not None:
        kp_img_path = build_output_name(
            dir_path,
            file_name,
            size=size,
            detector_name=detector_name,
            file_type='png',
            prefix=os.path.join('keypoint_images',
                                'kpts_{}_'.format(project_name)))
        save_keypoints_image(img_kp, kp_img_path)

    # Save heatmap
    if img_heatmap is not None:
        heat_img_path = build_output_name(
            dir_path,
            file_name,
            size=size,
            detector_name=detector_name,
            file_type='png',
            prefix=os.path.join('heatmap_images',
                                'heatmap_{}_'.format(project_name)))
        save_keypoints_image(img_heatmap, heat_img_path)
