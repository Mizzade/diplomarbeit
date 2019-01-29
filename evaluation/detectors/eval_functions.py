from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import json
import os
from tqdm import tqdm
import pickle
import util_functions as util
from Evaluater import Evaluater

def eval__num_kpts(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    """Returns pandas dataframe containing the number of found keypoints for
    each file in ev.file_list

    Arguments:
        ev {Evaluater} -- Evaluater object
        obj {Dict} -- The target object, where the output of this function will
        be saved to.

    Returns:
        pd.DataFrame -- DataFrame with keypoint list as index, and number of
        found keypoints as value column.
    """

    file_list = ev.file_system[ev.collection_name]['_file_list']

    output = pd.DataFrame(
        data=np.zeros((len(file_list), 1), dtype='int32'),
        index=file_list)

    # Open each file and number of rows, which is equal to number keypoints
    # that have been found. Insert number into output dataframe.
    for file_path in file_list:
        df = pd.read_csv(file_path, sep=',', delimiter='#')
        output.loc[file_path][0] = df.shape[0]

    return output

def eval__num_max_equal_kpts(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    file_list = ev.file_system[ev.collection_name][ev.set_name]

    output = pd.DataFrame(
        data=np.zeros((len(file_list), len(file_list)), dtype='int32'),
        index=file_list,
        columns=file_list
    )

    df_num_kpts = obj[ev.collection_name]['num_kpts']

    for i in file_list:
        for j in file_list:
            output.loc[i][j] = np.min([df_num_kpts.loc[i][0], df_num_kpts.loc[j][0]])

    return output



