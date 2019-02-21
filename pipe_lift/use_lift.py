from typing import List, Tuple, Dict, Any
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
from tqdm import tqdm
import copy

### This is from kp_tools.py in tf-lift folder
from six.moves import xrange
import scipy.ndimage
from scipy.linalg import lu_factor, lu_solve

# Keypoint List Structure Index Info
IDX_X, IDX_Y, IDX_SIZE, IDX_ANGLE, IDX_RESPONSE, IDX_OCTAVE = (
    0, 1, 2, 3, 4, 5)  # , IDX_CLASSID not used
IDX_a, IDX_b, IDX_c = (6, 7, 8)
# NOTE the row-major colon-major adaptation here
IDX_A0, IDX_A2, IDX_A1, IDX_A3 = (9, 10, 11, 12)
# # IDX_CLASSID for KAZE
# IDX_CLASSID = 13

KP_LIST_LEN = 13
# Note that the element in the IDX_SIZE field will be scale, not the opencv
# keypoint's size. As open cv uses diameters, we need to multiply this with two
# to get the opencv size.

# ------------------------------------------------------------------------
# Functions for Swapping between Opencv Kp and Our Kp List
# ------------------------------------------------------------------------

def kp_list_2_opencv_kp_list(kp_list):

    opencv_kp_list = []
    for kp in kp_list:
        opencv_kp = cv2.KeyPoint(x=kp[IDX_X],
                                 y=kp[IDX_Y],
                                 _size=kp[IDX_SIZE] * 2.0,
                                 _angle=kp[IDX_ANGLE],
                                 _response=kp[IDX_RESPONSE],
                                 _octave=np.int32(kp[IDX_OCTAVE]),
                                 # _class_id=np.int32(kp[IDX_CLASSID])
                                 )
        opencv_kp_list += [opencv_kp]

    return opencv_kp_list


def opencv_kp_list_2_kp_list(opencv_kp_list):

    # IMPORTANT: make sure this part corresponds to the one in
    # loadKpListFromTxt

    kp_list = []
    for opencv_kp in opencv_kp_list:
        kp = np.zeros((KP_LIST_LEN, ))
        kp[IDX_X] = opencv_kp.pt[0]
        kp[IDX_Y] = opencv_kp.pt[1]
        kp[IDX_SIZE] = opencv_kp.size * 0.5
        kp[IDX_ANGLE] = opencv_kp.angle
        kp[IDX_RESPONSE] = opencv_kp.response
        kp[IDX_OCTAVE] = opencv_kp.octave
        # Compute a,b,c for vgg affine
        kp[IDX_a] = 1. / (kp[IDX_SIZE]**2)
        kp[IDX_b] = 0.
        kp[IDX_c] = 1. / (kp[IDX_SIZE]**2)
        # Compute A0, A1, A2, A3 and update
        kp = update_affine(kp)
        # kp[IDX_CLASSID] = opencv_kp.class_id

        kp_list += [kp]

    return kp_list

def update_affine(kp):
    """Returns an updated version of the keypoint.

    Note
    ----
    This function should be applied only to individual keypoints, not a list.

    """

    # Compute A0, A1, A2, A3
    S = np.asarray([[kp[IDX_a], kp[IDX_b]], [kp[IDX_b], kp[IDX_c]]])
    invS = np.linalg.inv(S)
    a = np.sqrt(invS[0, 0])
    b = invS[0, 1] / max(a, 1e-18)
    A = np.asarray([[a, 0], [b, np.sqrt(max(invS[1, 1] - b**2, 0))]])

    # We need to rotate first!
    cos_val = np.cos(np.deg2rad(kp[IDX_ANGLE]))
    sin_val = np.sin(np.deg2rad(kp[IDX_ANGLE]))
    R = np.asarray([[cos_val, -sin_val], [sin_val, cos_val]])

    A = np.dot(A, R)

    kp[IDX_A0] = A[0, 0]
    kp[IDX_A1] = A[0, 1]
    kp[IDX_A2] = A[1, 0]
    kp[IDX_A3] = A[1, 1]

    return kp

def loadKpListFromTxt(kp_file_name):

    # Open keypoint file for read
    kp_file = open(kp_file_name, 'rb')

    # skip the first two lines
    kp_line = kp_file.readline()
    kp_line = kp_file.readline()

    kp_list = []
    num_elem = -1
    while True:
        # read a line from file
        kp_line = kp_file.readline()
        # check EOF
        if not kp_line:
            break
        # split read information
        kp_info = kp_line.split()
        parsed_kp_info = []
        for idx in xrange(len(kp_info)):
            parsed_kp_info += [float(kp_info[idx])]
        parsed_kp_info = np.asarray(parsed_kp_info)

        if num_elem == -1:
            num_elem = len(parsed_kp_info)
        else:
            assert num_elem == len(parsed_kp_info)

        # IMPORTANT: make sure this part corresponds to the one in
        # opencv_kp_list_2_kp_list

        # check if we have all the kp list info
        if len(parsed_kp_info) == 6:  # if we only have opencv info
            # Compute a,b,c for vgg affine
            a = 1. / (parsed_kp_info[IDX_SIZE]**2)
            b = 0.
            c = 1. / (parsed_kp_info[IDX_SIZE]**2)
            parsed_kp_info = np.concatenate((parsed_kp_info, [a, b, c]))

        if len(parsed_kp_info) == 9:  # if we don't have the Affine warp
            parsed_kp_info = np.concatenate((parsed_kp_info, np.zeros((4, ))))
            parsed_kp_info = update_affine(parsed_kp_info)

        # if len(parsed_kp_info) == 13:
        #     # add dummy class id
        #     parsed_kp_info = np.concatenate((parsed_kp_info, [0]))

        # make sure we have everything!
        assert len(parsed_kp_info) == KP_LIST_LEN

        kp_list += [parsed_kp_info]

    # Close keypoint file
    kp_file.close()

    return kp_list


def saveKpListToTxt(kp_list, orig_kp_file_name, kp_file_name):

    # first line KP_LIST_LEN to indicate we have the full
    kp_line = str(KP_LIST_LEN) + '\n'

    # Open keypoint file for write
    kp_file = open(kp_file_name, 'w')

    # write the first line
    kp_file.write(kp_line)

    # write the number of kp in second line
    kp_file.write('{}\n'.format(len(kp_list)))

    for kp in kp_list:

        # Make sure we have all info for kp
        assert len(kp) == KP_LIST_LEN

        # Form the string to write
        write_string = ""
        for kp_elem, _i in zip(kp, range(len(kp))):
            # if _i == IDX_OCTAVE or _i == IDX_CLASSID:
            if _i == IDX_OCTAVE:  # in case of the octave
                write_string += str(np.int32(kp_elem)) + " "
            else:
                write_string += str(kp_elem) + " "
        write_string += "\n"

        # Write the string
        kp_file.write(write_string)

    # Close keypoint file
    kp_file.close()

### kp_tools.py ends here.

# Deactivates compiler warnings for tensorflow cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def compute(image_file_path:str, config:Dict, model:Any) -> np.array:
    """
    Computes the descriptors for all keypoints in a keypoint .csv file.
    1) Check if keypoint file exisits.
    2) Load and scale image.
    2b) Save image in tmp dir.
    3) Load .csv file mit keypoints as cv2.KeyPoint list.
    4) ...and convert  it with correct format for LIFT.
    5) Save keypoint list to text in `tmp` dir as .txt
    6) Compute orientation
    7) compute descriptors
    8) Load descriptors as numpy
    """
    # Build paths
    lift_path = os.path.join(config['root_dir_lift'], 'tf-lift')
    path_tmp_img = os.path.join(config['tmp_dir_lift'], 'tmp_img.png')
    path_tmp_kpts = os.path.join(config['tmp_dir_lift'], 'tmp_kpts.txt')
    path_tmp_ori = os.path.join(config['tmp_dir_lift'], 'tmp_ori.txt')
    path_tmp_desc = os.path.join(config['tmp_dir_lift'], 'tmp_desc.h5')

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

    # 1) Check if keypoint file exists.
    if not os.path.isfile(keypoints_file_path):
        print('Could not find keypoints in path: {}\n.Skip'.format(keypoints_file_path))
        return None

    # 2) Load and scale image
    img = cv2.imread(image_file_path, 0)
    img = io_utils.smart_scale(img, config['max_size'], prevent_upscaling=config['prevent_upscaling']) if config['max_size'] is not None else img

    # 2b) Write image in tmp dir
    cv2.imwrite(path_tmp_img, img)

    # 3) Load .csv file mit keypoints as cv2.KeyPoint list.
    kpts_numpy = io_utils.get_keypoints_from_csv(keypoints_file_path)

    # Convert numpy array to List of cv2.KeyPoint list
    kpts_cv2 = io_utils.numpy_to_cv2_kp(kpts_numpy)

    # 4) Convert to LIFT format
    kpts_lift = opencv_kp_list_2_kp_list(kpts_cv2)

    # 5) Save keypoint list as .txt file for LIFT
    saveKpListToTxt(kpts_lift, None, path_tmp_kpts)

    try:
        # 6) Orientation
        subprocess.check_call(['python',
            'main.py',
            '--subtask=ori',
            '--test_img_file={}'.format(path_tmp_img),
            '--test_out_file={}'.format(path_tmp_ori),
            '--test_kp_file={}'.format(path_tmp_kpts)],
            cwd=lift_path)

        # 7) Descriptors
        subprocess.check_call(['python',
            'main.py',
            '--subtask=desc',
            '--test_img_file={}'.format(path_tmp_img),
            '--test_out_file={}'.format(path_tmp_desc),
            '--test_kp_file={}'.format(path_tmp_ori)],
            cwd=lift_path)
    except Exception:
        print('Could not compute descriptros for image {} at max_size {}. Skip.'.format(image_file_path, config['max_size']))
        return None

    # 8)
    f = h5py.File(path_tmp_desc, 'r')
    descriptors = np.array(list(f['descriptors']))

    return descriptors



def detect(image_path:str, config:Dict, detector:Any) -> None:
    """Detects keypoints for a given input image.
    Draws keypoints into the image.
    Returns keypoints, heatmap and image with keypoints.

    1) Load and smart scale the image.
    2) Save the resulting image in `tmp` folder.
    3) Subprocess call to tf-lift for keypoints. Save output text file in `tmp`.
    4) Load text file as as np.array with dimension [num_kp x 13]
    5) Convert keypoints to list of cv2.Keypoint.
    6) Draw list of cv2.KeyPoints into image.
    7) Return KeyPoint list, descriptors and image with KeyPoints.

    """
    lift_path = os.path.join(config['root_dir_lift'], 'tf-lift')

    # 1)
    img = cv2.imread(image_path, 0)
    img = io_utils.smart_scale(img, config['max_size'], prevent_upscaling=config['prevent_upscaling']) if config['max_size'] is not None else img

    # Build paths
    path_tmp_img = os.path.join(config['tmp_dir_lift'], 'tmp_img.png')
    path_tmp_kpts = os.path.join(config['tmp_dir_lift'], 'tmp_kpts.txt')
    path_tmp_ori = os.path.join(config['tmp_dir_lift'], 'tmp_ori.txt')
    path_tmp_desc = os.path.join(config['tmp_dir_lift'], 'tmp_desc.h5')

    # 2) Save image for lift to use
    cv2.imwrite(path_tmp_img, img)

    # 3) Keypoints
    num_kpts = 1000 if config['max_num_keypoints'] is None else config['max_num_keypoints']
    try:
        subprocess.check_call(['python',
            'main.py',
            '--subtask=kp',
            '--test_num_keypoint={}'.format(num_kpts),
            '--test_img_file={}'.format(path_tmp_img),
            '--test_out_file={}'.format(path_tmp_kpts)],
            cwd=lift_path)
    except Exception as e:
        print('Could not process image {} at max_size {}. Skip.'.format(image_path, config['max_size']))
        return (None, None, None)


    # 4) Load text file as list of N np.arrays of shape [1x13]
    kpts_numpy = loadKpListFromTxt(path_tmp_kpts)

    # 5) Convert to cv2.KeyPoint list
    kpts_cv2 = kp_list_2_opencv_kp_list(kpts_numpy)

    # 6) Draw keypoints in image
    img_kp = cv2.drawKeypoints(img, kpts_cv2, None)

    return (kpts_cv2, img_kp, None)

def main(argv: Tuple[str]) -> None:
    """Runs the LIFT model and saves the results.

    Arguments:
        argv {Tuple[str]} -- List of one parameters. There should be exactly
            one parameter - the path to the config file inside the tmp dir.
            This config file will be used to get all other information and
    """
    if len(argv) <= 0:
        raise RuntimeError("Missing argument <path_to_config_file>. Abort")

    with open(argv[0], 'rb') as src:
        config_file = pickle.load(src, encoding='utf-8')

    config, file_list = config_file
    model = None

    if config['task'] == 'keypoints':
        for file in tqdm(file_list):
            keypoints, keypoints_image, heatmap_image = detect(file, config, None)
            if keypoints is not None:
                io_utils.save_detector_output(
                    file,
                    config['detector_name'],
                    config,
                    keypoints,
                    keypoints_image,
                    heatmap_image)

    elif config['task'] == 'descriptors':
        for file in tqdm(file_list):
            descriptors = compute(file, config, model)
            if descriptors is not None:
                io_utils.save_descriptor_output(file, config, descriptors)

if __name__ == '__main__':
    argv = sys.argv[1:]
    main(argv)
