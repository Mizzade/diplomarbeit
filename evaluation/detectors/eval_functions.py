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
    i = ev.eval_config['i']
    j = ev.eval_config['j']

    try:
        output = obj[collection_name][set_name]['num_matching_kpts_with_e_{}'.format(epsilon)]
    except KeyError:
        output = pd.DataFrame(
            data=np.zeros((len(file_list), len(file_list)), dtype='int32'),
            index=file_list,
            columns=file_list)

    eps = np.sqrt(2 * epsilon * epsilon)

    kpts_1 = pd.read_csv(i, sep=',', comment='#').values
    kpts_2 = pd.read_csv(j, sep=',', comment='#').values
    max_num_matching_kpts = obj[collection_name][set_name]['max_num_matching_kpts'].loc[i][j]

    distance = np.sqrt(np.sum((kpts_1[:, np.newaxis] - kpts_2) ** 2, axis=2))
    distance[distance <= eps] = 1
    distance[distance > eps] = 0
    num_kpts = np.clip(np.sum(np.max(distance, axis=1)), 0, max_num_matching_kpts)

    output.loc[i][j] = num_kpts

    return output

def eval__perc_repeatability_for_image_pairs_with_e(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']
    file_list = ev.file_system[collection_name][set_name]

    max_num_matching_kpts = obj[collection_name][set_name]['max_num_matching_kpts'].values
    num_matching_kpts = obj[collection_name][set_name]['num_matching_kpts_with_e_{}'.format(epsilon)]

    data = np.divide(
        num_matching_kpts,
        max_num_matching_kpts,
        out=np.zeros_like(num_matching_kpts).astype('float32'),
        where=max_num_matching_kpts!=0)

    output = pd.DataFrame(
        data=data,
        index=file_list,
        columns=file_list)

    return output

def eval__avg_number_kpts_in_set(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    file_list = ev.file_system[collection_name][set_name]

    num_kpts = obj[collection_name]['num_kpts'].loc[file_list].values
    output = np.mean(num_kpts)

    return output


def eval__std_number_kpts_in_set(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    file_list = ev.file_system[collection_name][set_name]


    num_kpts = obj[collection_name]['num_kpts'].loc[file_list].values
    output = np.std(num_kpts)

    return output


def eval__avg_num_matching_kpts_in_set(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    num_matching_kpts = obj[collection_name][set_name]['num_matching_kpts_with_e_{}'.fomrat(epsilon)]
    output = np.mean(num_matching_kpts)

    return output


def eval__std_num_matching_kpts_in_set(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    num_matching_kpts = obj[collection_name][set_name]['num_matching_kpts_with_e_{}'.fomrat(epsilon)]
    output = np.std(num_matching_kpts)

    return output


def eval__avg_max_num_matching_kpts_in_set(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']

    max_num_matching_kpts = obj[collection_name][set_name]['max_num_matching_kpts']
    output = np.mean(max_num_matching_kpts)

    return output


def eval__std_max_num_matching_kpts_in_set(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']

    max_num_matching_kpts = obj[collection_name][set_name]['max_num_matching_kpts']
    output = np.std(max_num_matching_kpts)

    return output


def eval_avg_perc_matchting_kpts_in_set(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    perc_matching_kpts = obj[collection_name][set_name]['perc_matching_kpts_for_e_{}'.format(epsilon)]
    output = np.mean(perc_matching_kpts)

    return output

def eval_std_perc_matchting_kpts_in_set(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    perc_matching_kpts = obj[collection_name][set_name]['perc_matching_kpts_for_e_{}'.format(epsilon)]
    output = np.std(perc_matching_kpts)

    return output
