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

    collection_name = ev.eval_config['collection_name']
    file_list = ev.file_system[collection_name]['_file_list']

    output = pd.DataFrame(
        data=np.zeros((len(file_list), 1), dtype='int32'),
        index=file_list)

    # Open each file and number of rows, which is equal to number keypoints
    # that have been found. Insert number into output dataframe.
    for file_path in file_list:
        df = pd.read_csv(file_path, sep=',', comment='#')
        output.loc[file_path][0] = df.shape[0]

    return output

def eval__num_max_equal_kpts(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    file_list = ev.file_system[collection_name][set_name]

    output = pd.DataFrame(
        data=np.zeros((len(file_list), len(file_list)), dtype='int32'),
        index=file_list,
        columns=file_list
    )

    df_num_kpts = obj[collection_name]['num_kpts']

    for i in file_list:
        for j in file_list:
            output.loc[i][j] = np.min([df_num_kpts.loc[i][0], df_num_kpts.loc[j][0]])

    return output

def eval__num_matching_kpts_with_e(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    file_list = ev.file_system[collection_name][set_name]
    epsilon = ev.eval_config['epsilon']

    output = pd.DataFrame(
        data=np.zeros((len(file_list), len(file_list)), dtype='int32'),
        index=file_list,
        columns=file_list)

    eps = np.sqrt(2 * epsilon * epsilone)

    for i in file_list:
        kpts_1 = pd.read_csv(i, sep=',', comment='#').values
        for j in file_list:
            max_num_equal_kpts = obj[collection_name][set_name]['max_num_matching_kpts'].loc[i][j]
            kpts_2 = pd.read_csv(j, sep=',', comment='#').values

            distance = np.sqrt(np.sum((kpts_1[:, np.newaxis] - kpts_2) ** 2, axis=2))
            distance[distance <= eps] = 1
            distance[distance > eps] = 0
            num_kpts = np.clip(np.sum(np.max(distance, axis=1)), 0, max_num_equal_kpts)

            output.loc[i][j] = num_kpts

    return output




