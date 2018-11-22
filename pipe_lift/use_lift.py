from typing import List, Tuple, Any
import cv2
import numpy as np
import sys
import json
import pickle
import copyreg
import os
import io_utils
import subprocess
import h5py
import shutil

def compute(
    model: Any,
    image: str,
    size: int=None) -> Tuple[List[cv2.KeyPoint], np.array, np.array]:
    """Computes the keypoints and descriptors for a given input image.
    Draws keypoints into the image.
    Returns keypoints, descriptors and image with keypoints.

    Arguments:
        model Any | None -- The lift keypoint detector and descriptor.
        image {np.array} -- Path to the image.
        size {None} -- Maximal dimension of image. Default: None.

    Returns:
        Tuple[List[cv2.KeyPoint], np.array, np.array, None] -- Returns tuple (keypoints, descriptors, image with keypoints, image of heatmap).
    """
    """Wrapper function for tf-lift.
    1) Create a temporay folder `tmp` to save intermediate output.
    2a) Load and smart scale the image.
    2b) Save the resulting image in `tmp`.
    3a) Subprocess call to tf-lift für keypoints. Save output in `tmp`.
    3b) Subprocess call for tf-lift orientation. Save output in `tmp`.
    3c) Subprocess call for tf-lift descriptors Save output in `tmp`.
    4a) Load final .h5 output file from tf-lift, extract keypoints and
        descriptors.
    4b) Convert keypoints to list of cv2.Keypoint.
    4c) Draw list of cv2.KeyPoints into image.
    5) Return KeyPoint list, descriptors and image with KeyPoints.
    """

    base_path, _ =  os.path.splitext(os.path.abspath(__file__))
    base_path = os.sep.join(base_path.split(os.sep)[:-1])
    lift_path = os.path.join(base_path, 'tf-lift')
    # 1)
    path_tmp = os.path.join(base_path, 'tmp')
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp, exist_ok=True)

    # 2a) tf-lift wants a colored image.
    img = cv2.imread(image)
    img = io_utils.smart_scale(img, size, prevent_upscaling=True) if size is not None else img

    # Build paths
    path_tmp_img = os.path.join(path_tmp, 'tmp_img.png')
    path_tmp_kpts = os.path.join(path_tmp, 'tmp_kpts.txt')
    path_tmp_ori = os.path.join(path_tmp, 'tmp_ori.txt')
    path_tmp_desc = os.path.join(path_tmp, 'tmp_desc.h5')

    # 2b)
    cv2.imwrite(path_tmp_img, img)

    # 3a) Keypoints
    subprocess.check_call(['python',
        'main.py',
        '--subtask=kp',
        '--test_img_file={}'.format(path_tmp_img),
        '--test_out_file={}'.format(path_tmp_kpts)],
        cwd=lift_path)

    # 3b) Orientation
    subprocess.check_call(['python',
        'main.py',
        '--subtask=ori',
        '--test_img_file={}'.format(path_tmp_img),
        '--test_out_file={}'.format(path_tmp_ori),
        '--test_kp_file={}'.format(path_tmp_kpts)],
        cwd=lift_path)

    # 3c) Descriptors
    subprocess.check_call(['python',
        'main.py',
        '--subtask=desc',
        '--test_img_file={}'.format(path_tmp_img),
        '--test_out_file={}'.format(path_tmp_desc),
        '--test_kp_file={}'.format(path_tmp_ori)],
        cwd=lift_path)
    # 4a)
    #filename = os.path.join('tmp', 'desc.h5')
    f = h5py.File(path_tmp_desc, 'r')
    desc = np.array(list(f['descriptors']))
    keyp = np.array(list(f['keypoints']))

    # 4b)
    kp = [cv2.KeyPoint(x=x[0], y=x[1], _size=2*x[2], _angle=x[3],
        _response=x[4], _octave=np.int32(x[5]), _class_id=-1) for x in keyp]

    # 4c)
    img_kp = cv2.drawKeypoints(img, kp, None)

    # 5)
    return (kp, desc, img_kp, None)

def main(argv: List[str]) -> None:
    """Runs the LIFT model and saves the results.

    Arguments:
        argv {List[str]} -- List of parameters. The first paramters must be the
        path to the output directory where the result of this model will be
        saved. The second argument is a JSON-string, containing the list of all
        files that the model should work with.
    """

    assert isinstance(argv[0], str)
    assert isinstance(argv[1], str)
    assert isinstance(json.loads(argv[1]), list)

    project_name = 'lift'
    detector_name = 'LIFT'
    descriptor_name = 'LIFT'

    output_dir = argv[0]
    file_list = json.loads(argv[1])
    model = None
    size = 800

    for file in file_list:
        io_utils.save_output(file, compute(model, file, size), output_dir,
            detector_name, descriptor_name, project_name)

    # Clean up
    base_path, _ =  os.path.splitext(os.path.abspath(__file__))
    base_path = os.sep.join(base_path.split(os.sep)[:-1])
    path_tmp = os.path.join(base_path, 'tmp')
    shutil.rmtree(path_tmp, ignore_errors=True)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
