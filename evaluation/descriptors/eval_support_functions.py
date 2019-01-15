import numpy as np
import pandas as pd
import cv2
import collections
import pickle
import os
from typing import List, Dict, Tuple, Any

"""All support functions for the evaluations of descriptors."""

def get_file_names(data_dir: str, allowed_extensions: List[str], sort_files: bool=True) -> List[str]:
    """Returns list of file names inside `data_dir` that have extensions
    specified in `allowed_extensions`

    Arguments:
        data_dir {str} -- Path to data directory containing files.
        allowed_extensions {List[str]} -- Allowed extensions with dot, e.g. ['.png', '.jpg']

    Returns:
        file_names {List[str]} -- List of all file names in that directory.
    """

    file_names = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            f_name, f_ext = os.path.splitext(file)
            if f_ext.lower() in allowed_extensions:
                file_names.append(f_name)

    if sort_files:
        file_names = sorted(file_names)

    return file_names

def get_image_names(
    collection_name:str, 
    set_name:str, 
    config: Dict) -> List[str]:
    image_set_path = os.path.join(config['images_dir'], collection_name, set_name)
    image_names = get_file_names(image_set_path, config['allowed_extensions'])
    return image_names

def get_keypoint_files(
    model_name:str,
    detector_name:str,
    collection_name:str,
    set_name:str,
    image_names:List[str],
    config: Dict) -> List[str]:
    data_set_path = os.path.join(config['data_dir'], collection_name, set_name)
    kpts_file_names = [config['kpts_file_format'].format(model_name, file_name,
        detector_name, config['max_size']) for file_name in image_names]
    kpts_files = [os.path.join(data_set_path, 'keypoints', f) for f in kpts_file_names]
    return kpts_files

def get_descriptor_files(
    model_name:str,
    detector_name:str,
    descriptor_name:str,
    collection_name:str,
    set_name:str,
    image_names:List[str],
    config:Dict,
    ) -> List[str]:
    """Returns list containing the paths to descriptors files for given
    model parameters."""
    
    data_set_path = os.path.join(config['data_dir'], collection_name, set_name)
    desc_file_names = [config['desc_file_format'].format(model_name, file_name, detector_name, descriptor_name, config['max_size']) for file_name in image_names]
    desc_files = [os.path.join(data_set_path, 'descriptors', f) for f in desc_file_names]
    return desc_files

def update(d: Dict, u: Dict) -> Dict:
    """Updates values in dict d with the values in dict u. Returns updated
    dict d

    Arguments:
        d {Dict} -- The dictionary to be updated.
        u {Dict} -- A dictionary with new keys/values to be inserted into d.

    Returns:
        d {Dict} -- Updated version of d.
    """

    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def save_metrics(obj: Any, file_path: str) -> Dict:
    """Update and saves metrics. Tries to load an already existing metric object.
    Use an empty dict if it does not exists. Update new values inside the dict
    and save the resulting object again.

    Arguments:
        obj {Dict} -- A dictionary with new value to be updated in an already
            existing metric pickle file.
        file_path {str} -- Path to where to load and save metric pickle object.

    Returns:
        None
    """

    try:
        with open(file_path, 'rb') as src:
            data = pickle.load(src, encoding='utf-8')
    except FileNotFoundError:
            data = {}

    update(data, obj)

    with open(file_path, 'wb') as dst:
        pickle.dump(data, dst, protocol=pickle.HIGHEST_PROTOCOL)

    return data

def get_flann():
    # CREATE FLANN 
    # FLANN parameters and FLANN object
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    return flann

def compute_distances_kpts_to_epilines(points_i, points_j, F:np.array) -> np.array:

    # Epipolar lines in image I of the points in image J
    lines_i = cv2.computeCorrespondEpilines(points_j.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    # Epipolar lines in image J of the points in image I
    lines_j = cv2.computeCorrespondEpilines(points_i.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    dist = []
    for k in range(points_i.shape[0]):
        # Params for image i
        xi, yi = points_i[k]
        ai, bi, ci = lines_i[k]

        # Params for image j
        xj, yj = points_j[k]
        aj, bj, cj = lines_j[k]

        di = np.abs(ai*xi + bi*yi + ci) / np.sqrt(ai*ai + bi*bi)
        dj = np.abs(aj*xj + bj*yj + cj) / np.sqrt(aj*aj + bj*bj)

        dist.append((di, dj))

    dist = np.array(dist)
    return dist

def apply_ratio_test_to_matches(matches:List, kpts_i:pd.DataFrame, kpts_j:pd.DataFrame) -> (List, List, List, List):
    good = []
    pts_i = []
    pts_j = []
    match_ids = []

    # ratio test as per Lowe's paper
    for idx,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts_j.append(tuple(kpts_j.loc[m.trainIdx, ['x', 'y']]))
            pts_i.append(tuple(kpts_i.loc[m.queryIdx, ['x', 'y']]))
            match_ids.append((m.trainIdx, m.queryIdx))
    return good, pts_i, pts_j, match_ids

def compute_fundamental_matrix(pts_i:np.array, pts_j:np.array, method=cv2.FM_RANSAC) -> (np.array, np.array):
    pts_i = np.int32(pts_i)
    pts_j = np.int32(pts_j)
    F, mask = cv2.findFundamentalMat(pts_i, pts_j, method)

    return F, mask



