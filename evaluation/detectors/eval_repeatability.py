import numpy as np
import pandas as pd
import cv2
import os
from typing import List, Tuple, Any

def get_file_names(data_dir: str, allowed_extensions: List[str]) -> List[str]:
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
    return file_names

def load_kpts_csv_file(file_path: str, delimiter: str=',', skiprows: int=3, usecols: List[int]=(0, 1), names: List[str]=['x', 'y']) -> pd.DataFrame:
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


def find_repeatable_keypoints(kpts_files: List[str], output_dir: str, epsilon: int=1) -> pd.DataFrame:
    """Lists in a binary Pandas DataFrame which keypoints from the first image
    in `kpts_files` has been found in another image of `kpts_files`.. The last
    column in the output DataFrame marks 1 if the keypoint has been found all
    images, otherwise 0.

    Arguments:
        kpts_files {List[str]} -- File names of the keypoints .csv files.
        output_dir {str} -- Path of the directory, containing the keypoint .csv
        files.

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

    df_source = load_kpts_csv_file(os.path.join(output_dir, kpts_files[0]))
    df_result = pd.DataFrame(index=df_source.index)

    for kp_file in kpts_files[1:]:
        df_query = load_kpts_csv_file(os.path.join(output_dir, kp_file))
        df_result[kp_file] = df_query \
            .apply(lambda s: (np.abs(df_source.x - s[0]) <= epsilon) &
                             (np.abs(df_source.y - s[1]) <= epsilon), axis=1) \
                                .astype('int').max(axis=0)


    # Add column describing, if the keypoint was found in all image (1) or not (0).
    df_result['is_repeatable'] = (df_result.sum(axis=1) == len(kpts_files) -1).astype('int')

    return df_result

def get_repeatability_for_n_images(df_rp_kpts: pd.DataFrame) -> List[int]:
    """Returns a list of number of repeatable keypoints. The i-th index in the
    list is the number of repeatable keypoints for i+1 images.

    Arguments:
        df_rp_kpts {pd.DataFrame} -- DataFrame returned from `find_repeatable_keypoints`.

    Returns:
        List[int] -- Number of repeatable keypoints over N images.
    """

    num_repeatable_keypoints = [df_rp_kpts.shape[0]]

    for i in range(df_rp_kpts.shape[1] - 1):
        num_repeatable_keypoints.append((df_rp_kpts.iloc[:, : i+1].sum(axis=1) == i+1).astype('int').sum(axis=0))

    return num_repeatable_keypoints

