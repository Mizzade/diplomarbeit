from typing import List, Tuple, Any, Dict
import cv2
import sys
import os
import numpy as np
import shutil

def create_dir(path: str):
    if path is not None and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def remove_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def build_output_file_name(
    image_name:str,
    max_size:int=None,
    extension:str=None) -> str:
    """Returns the output file name of a given image.
    Output format is: <prefix><image_name>_<max_size>.<extension>"""
    _m = '_' + str(max_size) if max_size else ''
    _e = 'csv' if extension is None else extension
    return '{}{}.{}'.format(image_name, _m, _e)

def build_output_dir_path(
    output_dir:str,
    collection_name:str,
    set_name:str,
    output_type:str,
    detector_name:str,
    descriptor_name:str=None) -> str:
    """Returns the absolute path depending on the used model and type of data
    that should be saved. OUTPUT_TYPE must be one of
    Output format is:
    <output_dir>/<collection_name>/<set_name>/<output_type>/<model_name>"""

    assert output_type in ['keypoints', 'descriptors', 'keypoint_images', 'heatmap_images']
    _l = [x for x in [output_dir, collection_name, set_name, output_type, descriptor_name, detector_name] if x is not None]
    return os.path.join(*_l)

def build_output_path(
    output_dir:str,
    collection_name:str,
    set_name:str,
    output_type:str,
    detector_name:str,
    image_name:str,
    descriptor_name:str=None,
    max_size:int=None,
    extension:str=None) -> str:
    _d = build_output_dir_path(output_dir, collection_name, set_name, output_type, detector_name, descriptor_name=descriptor_name)
    _f = build_output_file_name(image_name, max_size, extension)

    return os.path.join(_d, _f)

def get_path_components(file_path: str) -> (str, str, str, str):
    """Returns the collection name, set name , file name and its extension for
    any given file path. The schema is:
    file_path = .../<collection_name>/<set_name>/<file_name>.<extensions>
    """
    _base_path, extension = os.path.splitext(file_path)
    collection_name, set_name, file_name = _base_path.split(os.sep)[-3:]

    return collection_name, set_name, file_name, extension

def get_keypoints_from_csv(
    file_path:str,
    dtype:Any=int,
    comments:str='#',
    delimiter:str=',') -> np.array:
    """Loads a .csv file containing keypoints and returns the corresponding
    numpy array."""
    #return np.loadtxt(file_path, dtype=dtype, comments=comments, delimiter=delimiter)
    return np.loadtxt(open(file_path, 'rb'), delimiter=delimiter, comments=comments)
    #return pd.read_csv('file_path', comment=comment, delimiter=delimiter)

def save_detector_output(
    file_path:str,
    model_name:str,
    config:Dict,
    keypoints:List[cv2.KeyPoint],
    keypoints_image:np.array,
    heatmap_image:np.array=None) -> None:
    image_shape = keypoints_image.shape
    collection_name, set_name, file_name, _ = get_path_components(file_path)

    # Keypoints
    output_keypoints_path = build_output_path(config['output_dir'],
        collection_name, set_name, 'keypoints', model_name, file_name,
        max_size=config['max_size'])
    save_keypoints_list(keypoints, output_keypoints_path, image_shape, verbose=config['verbose'])

    # Image of keypoints
    output_keypoints_image_path = build_output_path(config['output_dir'],
        collection_name, set_name, 'keypoint_images', model_name, file_name,
        max_size=config['max_size'], extension='png')
    save_keypoints_image(keypoints_image, output_keypoints_image_path, verbose=config['verbose'])

    # Heatmap
    if heatmap_image is not None:
        output_heatmap_path = build_output_path(config['output_dir'],
            collection_name, set_name, 'heatmap_images', model_name, file_name,
            max_size=config['max_size'], extension='png')
        save_keypoints_image(heatmap_image, output_heatmap_path, config['verbose'])

def save_descriptor_output( file_path:str, config:Dict, descriptors:np.array) -> None:
    collection_name, set_name, image_name, _ = get_path_components(file_path)
    output_descriptor_path = build_output_path(
        config['output_dir'],
        collection_name,
        set_name,
        'descriptors',
        config['detector_name'],
        image_name,
        descriptor_name=config['descriptor_name'],
        max_size=config['max_size'])
    save_descriptors(descriptors, output_descriptor_path, verbose=config['verbose'])

def numpy_to_cv2_kp(kpts_numpy:np.array) -> List[cv2.KeyPoint]:
    """Converts a numpy array containing keypoints to a list of cv2.KeyPoint
    elements."""
    if kpts_numpy.shape[1] == 7:
        return _numpy_to_cv2_kp(kpts_numpy)
    elif kpts_numpy.shape[1] == 13:
        return _numpy_to_cv2_kp_lift(kpts_numpy)
    else:
        raise Exception('Could not convert numpy array to cv2.KeyPoint list.')
    kpts_cv2 = []

def _numpy_to_cv2_kp(kpts_numpy:np.array) -> List[cv2.KeyPoint]:
    """Converts a numpy array containing keypoints to a list of opencv
    KeyPoint elements. The array must have 7 columns corresponding to the
    7 parameters cv2.KeyPoint has."""
    kpts_cv2 = []
    for kp in kpts_numpy:
        kpts_cv2.append(cv2.KeyPoint(
            x=kp[0],
            y=kp[1],
            _size=kp[2],
            _angle=kp[3],
            _response=kp[4],
            _octave=np.int32(kp[5]),
            _class_id=np.int32(kp[6])
        ))

    return kpts_cv2

def _numpy_to_cv2_kp_lift(kpts_numpy:np.array) -> List[cv2.KeyPoint]:
    """Converts a numpy array containing the keypoints generated by LIFT
    detector to opencv KeyPoint list. LIFT keypoints come with 13 columns and
    need special treatment to be converted to cv2.KeyPoint
    """
    kpts_cv2 = []
    for kp in kpts_numpy:
        kpts_cv2.append(cv2.KeyPoint(
            x=kp[0],
            y=kp[1],
            _size=kp[2] * 2.0,
            _angle=kp[3],
            _response=kp[4],
            _octave=np.int32(kp[5])
        ))

    return kpts_cv2


def save_keypoints_list(kpts: List[cv2.KeyPoint], file_name: str, image_shape: np.array, verbose:bool=False) -> None:
    _dirname = os.path.dirname(file_name)
    create_dir(_dirname)

    if verbose:
        print('Save keypoint list into {}'.format(file_name))

    _a = np.array([[*x.pt, x.size, x.angle, x.response, x.octave, x.class_id] for x in kpts])
    np.savetxt(file_name,_a, delimiter=',',
        header='height, width, number of rows, number of columns\n{}, {}, {}, {}\nx, y, size, angle, response, octave, class_id' \
            .format(image_shape[0], image_shape[1], _a.shape[0], _a.shape[1]))

def save_descriptors(desc: np.array, file_name: str, verbose:bool=False) -> None:
    _dirname = os.path.dirname(file_name)
    create_dir(_dirname)

    if verbose:
        print('Save descriptor list into {}'.format(file_name))

    np.savetxt(file_name, desc, delimiter=',',
        header='{}, {}'.format(desc.shape[0], desc.shape[1]))

def save_keypoints_image(image: np.array, file_name: str, verbose:bool=False) -> None:
    _dirname = os.path.dirname(file_name)
    create_dir(_dirname)

    if verbose:
        print('Save keypoint image/heatmap image into {}'.format(file_name))

    cv2.imwrite(file_name, image)

def save_patches_list(patches: List[np.array], file_name:str, verbose:bool=False) -> None:
    _dirname = os.path.dirname(file_name)
    create_dir(_dirname)

    if verbose:
        print('Save patches list into {}'.format(file_name))

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

def build_output_name(dir_path: str, image_name: str, detector_name: str='',
    prefix='', descriptor_name: str='', size: int=None, file_type: str='csv') -> str:
    # TODO assert that at least detector_name or descriptor_name is set
    if not detector_name and not descriptor_name:
        sys.exit('Error: `detector_name` or `descriptor_name` must be set')


    _list = [x for x in [prefix, image_name, detector_name, descriptor_name] if x]

    if size is not None:
        _list.append(str(size))

    return os.path.join(dir_path, '_'.join(_list) + '.{}'.format(file_type))
