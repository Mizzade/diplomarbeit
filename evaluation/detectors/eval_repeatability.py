import numpy as np
import pandas as pd
import cv2
import os
import pickle
import sys
import collections
from itertools import combinations
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import config_repeatability as cfg_rep

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

def load_kpts_csv_file(
    file_path:str,
    delimiter:str=',',
    skiprows:int=3,
    usecols:List[int]=(0, 1),
    names:List[str]=['x', 'y']) -> pd.DataFrame:
    """Loads a csv file for keypoints and returns is as a Pandas Dataframe. This
    is a convenience wrapper function, where the usual paramters are already set as default.
    The given parameters a part of the Pandas function `read_csv`.

    Arguments:
        file_path {str} -- Path to .csv file containing the keypoints.

    Keyword Arguments:
        delimiter {str} -- Delimiter string. (default: {','})
        skiprows {int} -- Skip first n rows. (default: {3})
        usecols {List[int]} -- Specify which columns to use in .csv file.  (default: {(0, 1)})
        names {List[str]} -- Custom names for columns (default: {['x', 'y']})

    Returns:
        df_keypoints {pd.DataFrame} -- DataFrame containing the keypoints in the .csv file.
    """
    return pd.read_csv(file_path,
                        delimiter=delimiter,
                        skiprows=skiprows,
                        usecols=usecols,
                        names=names)


def find_repeatable_keypoints(
    kpts_files: List[str],
    epsilon: int=1) -> pd.DataFrame:
    """Lists in a binary Pandas DataFrame which keypoints from the first image
    in `kpts_files` has been found in another image of `kpts_files`. The last
    column in the output DataFrame marks 1 if the keypoint has been found all
    images, otherwise 0.

    Arguments:
        kpts_files {List[str]} -- Kyepoint .csv files.

    Keyword Arguments:
        epsilon {int} -- Max distance in pixel, for which a pair of keypoints
        is considered to be still in the same position of the image. (default: {1})

    Returns:
        df_result {pd.DataFrame} -- Binaray DataFrame. Each row is a key point of
        the first image in `kpts_files`. Each column is labeled as the name of
        of the .csv file, with which to compare said keypoints. The last column in
        `df_result` is called `is_repeatable` and is 1, if the keypoint in that
        row has been found in all other files as well.
    """

    df_source = load_kpts_csv_file(kpts_files[0])
    df_result = pd.DataFrame(index=df_source.index)

    print('\tstart find_repeatable_keypoints. source is ', kpts_files[0])

    for kp_file in kpts_files:
        df_query = load_kpts_csv_file(kp_file)
        df_result[kp_file] = df_query \
            .apply(lambda s: (np.abs(df_source.x - s[0]) <= epsilon) &
                             (np.abs(df_source.y - s[1]) <= epsilon), axis=1) \
                                .astype('int').max(axis=0)


    # Add column describing, if the keypoint was found in all image (1) or not (0).
    df_result['repeatable'] = (df_result.sum(axis=1) == len(kpts_files) -1).astype('int')

    return df_result

def get_repeatability_for_n_images(df_rp_kpts: pd.DataFrame) -> np.array:
    """Returns a list of number of repeatable keypoints. The i-th index in the
    list is the number of repeatable keypoints for i+1 images.

    Arguments:
        df_rp_kpts {pd.DataFrame} -- DataFrame returned from `find_repeatable_keypoints`.

    Returns:
        List[int] -- Number of repeatable keypoints over N images.
    """
    print('Started get_repeatability_for_n_images')

    num_repeatable_keypoints = [df_rp_kpts.shape[0]]

    for i in range(df_rp_kpts.shape[1] - 1):
        num_repeatable_keypoints.append((df_rp_kpts.iloc[:, : i+1].sum(axis=1) == i+1).astype('int').sum(axis=0))

    return np.array(num_repeatable_keypoints).astype('int')

def get_number_of_keypoints_per_image(kpts_files:List[str]) -> np.array:
    """Returns the number of found keypoints for each keypoint file in `kpts_files`
    as an np.array.

    Arguments:
        kpts_files {List[str]} -- List of all files containing keypoints.

    Returns:
        num_kp_per_image {np.array} -- Each element i describes the number of found
        keypoints for the i-th keypoint file in `kpts_files`.
    """

    num_kp_per_image = []
    for kp_file in kpts_files:
        kpts = load_kpts_csv_file(kp_file)
        num_kp_per_image.append(kpts.shape[0])

    return np.array(num_kp_per_image)

def get_num_repeatable_keypoints_for_epsiolon(df_source: pd.DataFrame, df_query: pd.DataFrame, epsilon:int=0):
    return df_query \
        .apply(lambda s: (np.abs(df_source.x - s[0]) <= epsilon) &
                            (np.abs(df_source.y - s[1]) <= epsilon), axis=1) \
        .astype('int') \
        .max(axis=0) \
        .sum()

def get_number_of_keypoints_for_all_image_pairs(kpts_files:List[str], epsilon:int=0) -> np.array:
    _i = None

    num_files = len(kpts_files)
    df_result = pd.DataFrame(np.zeros((num_files, num_files)))
    df_source = None
    df_query = None

    for i, j in combinations(range(num_files), 2):

        # Only load source df, if the first element in combination changes.
        # Prevents reloading the same file multiple times.
        if _i != i:
            df_source = load_kpts_csv_file(kpts_files[i])
            _i = i
            df_result.iloc[i, i] = df_source.shape[0]

        df_query = load_kpts_csv_file(kpts_files[j])

        df_result.iloc[i ,j] = get_num_repeatable_keypoints_for_epsiolon(df_source, df_query, epsilon=epsilon)
        df_result.iloc[j ,i] = get_num_repeatable_keypoints_for_epsiolon(df_query, df_source, epsilon=epsilon)

    # Finally add the number of keypoints of the last file.
    df_result.iloc[num_files - 1, num_files -1] = df_query.shape[0]

    return df_result.values

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

def get_metrics_for_epsilon(
    model_name:str,
    detector_name:str,
    collection_name:str,
    set_name:str,
    epsilon:int,
    kpts_files:List[str],
    config: Dict) -> Dict:
    print('Get epsilon metric for model: {}, set: {} and eps: {}.'.format(model_name, set_name, epsilon))
    metrics = {}

    if config['eval_set__num_repeatable_kpts'] or \
        config['eval_set__idx_repeatable_kpts'] or \
        config['eval_set__cum_repeatable_kpts']:
        df_rkpts = find_repeatable_keypoints(kpts_files, epsilon)

    if config['eval_set__num_repeatable_kpts']:
        metrics['num_repeatable_kpts'] = df_rkpts['repeatable'].sum()

    if config['eval_set__idx_repeatable_kpts']:
        metrics['idx_repeatable_kpts'] = list(df_rkpts[df_rkpts['repeatable'] == 1].index)

    if config['eval_set__idx_repeatable_kpts']:
        metrics['cum_repeatable_kpts'] = get_repeatability_for_n_images(df_rkpts)

    if config['eval_set__repeatable_kpts_image_pairs']:
        metrics['repeatable_kpts_image_pairs'] = \
            get_number_of_keypoints_for_all_image_pairs(kpts_files, epsilon)

    return metrics

def get_metric_for_set(
    model_name:str,
    detector_name:str,
    collection_name:str,
    set_name:str,
    config: Dict) -> Dict:

    print('Get metric for model {} and set {}.'.format(model_name, set_name))
    metrics = {}
    metrics['epsilon'] = {}

    image_names = get_image_names(collection_name, set_name, config)
    kpts_files = get_keypoint_files(model_name, detector_name, collection_name,
        set_name, image_names, config)

    if config['eval_set__image_names']:
        metrics['image_names'] = image_names

    # Number of keypoints found in each image
    if config['eval_set__num_kpts_per_image'] or \
        config['eval_set__num_kpts_per_image_avg'] or \
        config['eval_set__num_kpts_per_image_std']:
        num_kpts_per_image = get_number_of_keypoints_per_image(kpts_files)

    if config['eval_set__num_kpts_per_image']:
        metrics['num_kpts_per_image'] = num_kpts_per_image

    if config['eval_set__num_kpts_per_image_avg']:
        metrics['num_kpts_per_image_avg'] = num_kpts_per_image.mean()

    if config['eval_set__num_kpts_per_image_std']:
        metrics['num_kpts_per_image_std'] = num_kpts_per_image.std()

    # Get repeatable Keypoints
    for eps in config['epsilons']:
        metrics['epsilon'][eps] = get_metrics_for_epsilon(
            model_name, detector_name, collection_name, set_name, eps, kpts_files, config)

    return metrics

def get_metrics_for_collection(
    model_name:str,
    detector_name:str,
    collection_name:str,
    config:Dict) -> Dict:
    metrics = {}
    for set_name in config['set_names']:
        metrics[set_name] = get_metric_for_set(model_name, detector_name,
            collection_name, set_name, config)
    return metrics

def get_metrics(
    model_name:str,
    detector_name:str,
    config: Dict) -> Dict:
    metrics = {}
    for collection_name in config['collection_names']:
        metrics[collection_name] = get_metrics_for_collection(model_name,
            detector_name, collection_name, config)
    return metrics

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

def main(config: Dict):
    for model_name, detector_name in tqdm(list(zip(config['model_names'], config['detector_names']))):
        dst_name = ''.join([config['output_file_prefix'], model_name, '.pkl'])
        dst_file_path = os.path.join(config['output_dir'], dst_name)
        metrics = get_metrics(model_name, detector_name, config)
        save_metrics(metrics, dst_file_path)

if __name__ == '__main__':
    argv = sys.argv[1:]
    config = cfg_rep.get_config(argv)
    if config['dry']:
        print(config)
    else:
        main(config)
