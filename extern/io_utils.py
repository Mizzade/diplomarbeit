from typing import List, Tuple, Any
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

def save_output(
    file_list: List[str],
    output: List[Tuple[List[cv2.KeyPoint], np.array, np.array, np.array]],
    output_dir: str,
    detector_name,
    descriptor_name,
    project_name) -> None:
    """Save the output of this model inside the `output_dir`

    Arguments:
        file_list {List[str]} -- List of all file paths.
        output {List[Tuple[List[cv2.KeyPoint], np.array, np.array]]} -- The output of this model.
        Expects a 4-tuple, containing a list of of keypoints, descriptors,
        image with keypoints and a heatmap. If a model does not provide one of
        those four elements, use a list of None instead.
        output_dir {str} -- Path to the output directory
        detector_name {str} -- Name of the used detector
        descriptor_name {str} -- Name of the used descriptor
        project_name {str} -- Name of project.
    """

    for file_path, (kpts, desc, img_kp, img_heatmap) in zip (file_list, output):
        set_name, file_name, _ = get_setName_fileName_extension(file_path)
        dir_path = os.path.join(output_dir, set_name)

        # Save list of keypoints.
        if kpts is not None:
            kp_path = build_output_name(
                dir_path,
                file_name,
                detector_name=detector_name,
                prefix=os.path.join('keypoints',
                                    'kpts_{}_'.format(project_name)))
            save_keypoints_list(kpts, kp_path, img_kp.shape)

        # Save descriptors
        if desc is not None:
            desc_path = build_output_name(
                dir_path,
                file_name,
                descriptor_name=descriptor_name,
                prefix=os.path.join('descriptors',
                                    'desc_{}_'.format(project_name)))
            save_descriptors(desc, desc_path)

        # Save keypoint image
        if img_kp is not None:
            kp_img_path = build_output_name(
                dir_path,
                file_name,
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
                detector_name=detector_name,
                file_type='png',
                prefix=os.path.join('heatmap_images',
                                    'heatmap_{}_'.format(project_name)))
            save_keypoints_image(img_heatmap, heat_img_path)

