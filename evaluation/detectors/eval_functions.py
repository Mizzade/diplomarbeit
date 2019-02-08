from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import json
import os
from tqdm import tqdm
import pickle
from Evaluater import Evaluater

def store_meta_information(ev:Evaluater, obj:Dict) -> Dict:
    """Stores meta information, like config and file_system object etc.

    Arguments:
        ev {Evaluater} -- The evaluater object. Allows access to config,
        file_system and all Evaluter properites.
        obj {Dict} -- The target object wherein to save the output of this function.
        Helpful if you want to access already computed elements.

    Returns:
        Dict -- Element to store wihtin `obj`.
    """
    output = {
        'config': ev.config,
        'file_system': ev.file_system
    }

    return output


def eval_image__num_kpts(ev:Evaluater, obj:Dict) -> pd.DataFrame:
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

def eval_imagepair__num_max_matching_kpts(ev:Evaluater, obj:Dict) -> pd.DataFrame:
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

def eval__num_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    file_list = ev.file_system[collection_name][set_name]
    epsilon = ev.eval_config['epsilon']
    i = ev.eval_config['i']
    j = ev.eval_config['j']

    try:
        output = obj[collection_name][set_name]['num_matching_kpts_for_e_{}'.format(epsilon)]
    except KeyError:
        output = pd.DataFrame(
            data=np.zeros((len(file_list), len(file_list)), dtype='int32'),
            index=file_list,
            columns=file_list)

    eps = np.sqrt(2 * epsilon * epsilon)

    kpts_1 = pd.read_csv(i, sep=',', comment='#', usecols=[0, 1], names=['x', 'y']).values
    kpts_2 = pd.read_csv(j, sep=',', comment='#', usecols=[0, 1], names=['x', 'y']).values
    max_num_matching_kpts = obj[collection_name][set_name]['max_num_matching_kpts'].loc[i][j]

    distance = np.sqrt(np.sum((kpts_1[:, np.newaxis] - kpts_2) ** 2, axis=2))
    distance[distance <= eps] = 1
    distance[distance > eps] = 0
    num_kpts = np.clip(np.sum(np.max(distance, axis=1)), 0, max_num_matching_kpts).astype('int32')
    output.loc[i][j] = num_kpts

    return output

def eval_imagepair__perc_matching_keypoints_for_e(ev:Evaluater, obj:Dict) -> pd.DataFrame:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']
    file_list = ev.file_system[collection_name][set_name]

    max_num_matching_kpts = obj[collection_name][set_name]['max_num_matching_kpts'].values
    num_matching_kpts = obj[collection_name][set_name]['num_matching_kpts_for_e_{}'.format(epsilon)]

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

def eval_set__avg_num_kpts(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    file_list = ev.file_system[collection_name][set_name]

    num_kpts = obj[collection_name]['num_kpts'].loc[file_list].values
    output = np.mean(num_kpts)

    return output

def eval_set__std_num_kpts(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    file_list = ev.file_system[collection_name][set_name]

    num_kpts = obj[collection_name]['num_kpts'].loc[file_list].values
    output = np.std(num_kpts)

    return output

def eval_set__stats_num_kpts(ev:Evaluater, obj:Dict) -> Dict:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    config = ev.config
    fs = ev.file_system

    y = obj[collection_name]['num_kpts'].loc[fs[collection_name][set_name]].values.flatten()
    avg = obj[collection_name][set_name]['avg_num_kpts']
    std = obj[collection_name][set_name]['std_num_kpts']

    idx_min = np.argmin(y)
    idx_max = np.argmax(y)

    val_min = np.min(y)
    val_max = np.max(y)

    lt_std = (y < avg - std)
    gt_std = (y > avg + std)
    condition = lt_std | gt_std

    idx_extrema = np.where(condition)[0]
    val_extrema = y[condition].flatten()
    num_extrema = len(val_extrema)

    idx_extrema_lt_std = np.where(lt_std)[0]
    val_extrema_lt_std = y[lt_std]
    num_extrema_lt_std = len(val_extrema_lt_std)

    idx_extrema_gt_std = np.where(gt_std)[0]
    val_extrema_gt_std = y[gt_std]
    num_extrema_gt_std = len(val_extrema_gt_std)


    output = {
        'y': y,
        'avg': avg,
        'std': std,
        'idx_min': idx_min,
        'idx_max': idx_max,
        'val_min': val_min,
        'val_max': val_max,
        'idx_extrema': idx_extrema,
        'val_extrema': val_extrema,
        'num_extrema': num_extrema,
        'idx_extrema_lt_std': idx_extrema_lt_std,
        'val_extrema_lt_std': val_extrema_lt_std,
        'num_extrema_lt_std': num_extrema_lt_std,
        'idx_extrema_gt_std': idx_extrema_gt_std,
        'val_extrema_gt_std': val_extrema_gt_std,
        'num_extrema_gt_std': num_extrema_gt_std
    }

    return output

def eval_set__avg_num_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    num_matching_kpts = obj[collection_name][set_name]['num_matching_kpts_for_e_{}'.format(epsilon)].values
    output = np.mean(num_matching_kpts)

    return output

def eval_set__std_num_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    num_matching_kpts = obj[collection_name][set_name]['num_matching_kpts_for_e_{}'.format(epsilon)].values
    output = np.std(num_matching_kpts)

    return output

def eval_set__avg_max_num_matching_kpts(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']

    max_num_matching_kpts = obj[collection_name][set_name]['max_num_matching_kpts'].values
    output = np.mean(max_num_matching_kpts)

    return output

def eval_set__std_max_num_matching_kpts(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']

    max_num_matching_kpts = obj[collection_name][set_name]['max_num_matching_kpts'].values
    output = np.std(max_num_matching_kpts)

    return output

def eval_set__avg_perc_matchting_kpts_for_e(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    perc_matching_kpts = obj[collection_name][set_name]['perc_matching_kpts_for_e_{}'.format(epsilon)].values
    output = np.mean(perc_matching_kpts)

    return output

def eval_set__std_perc_matchting_kpts_for_e(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    perc_matching_kpts = obj[collection_name][set_name]['perc_matching_kpts_for_e_{}'.format(epsilon)].values
    output = np.std(perc_matching_kpts)

    return output

def eval_set__stats_perc_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> Dict:
    collection_name = ev.eval_config['collection_name']
    set_name = ev.eval_config['set_name']
    epsilon = ev.eval_config['epsilon']

    heatmap = obj[collection_name][set_name]['perc_matching_kpts_for_e_{}'.format(epsilon)].values

    # create a mask, that hides diagonal elements
    mask = np.ones(heatmap.shape).astype('int')
    np.fill_diagonal(mask, 0)

    # Remove the diagonal elements within the square array to create
    # an new matrix from NxN to Nx(N-1)
    rows = heatmap[np.where(mask)].reshape(heatmap.shape[0], heatmap.shape[1]-1)

    # Create a new array containing the mean value of each row. (1, N-1).
    # And compute the mean value of that too.
    y = rows.mean(axis=1)
    avg = y.mean()
    std = y.std()

    val_min = np.min(y)
    val_max = np.max(y)
    idx_min = np.argmin(y)
    idx_max = np.argmax(y)


    lt_std = y < avg - std
    gt_std = y > avg +std
    condition = (lt_std) | (gt_std)
    val_extrema = y[condition].flatten()
    idx_extrema = np.where(condition)[0]
    num_extrema = len(val_extrema)

    val_extrema_lt_std = y[lt_std].flatten()
    idx_extrema_lt_std = np.where(lt_std)[0]
    num_extrema_lt_std = len(val_extrema_lt_std)

    val_extrema_gt_std = y[gt_std].flatten()
    idx_extrema_gt_std = np.where(gt_std)[0]
    num_extrema_gt_std = len(val_extrema_gt_std)

    output = {
        '_description': "Average repeatability of keypoints of image i in all other images j != i.",
        'y': y,
        'rows': rows,
        'avg': avg,
        'std': std,
        'val_min': val_min,
        'val_max': val_max,
        'idx_min': idx_min,
        'idx_max': idx_max,
        'val_extrema': val_extrema,
        'idx_extrema': idx_extrema,
        'num_extrema': num_extrema,
        'val_extrema_lt_std': val_extrema_lt_std,
        'idx_extrema_lt_std': idx_extrema_lt_std,
        'num_extrema_lt_std': num_extrema_lt_std,
        'val_extrema_gt_std': val_extrema_gt_std,
        'idx_extrema_gt_std': idx_extrema_gt_std,
        'num_extrema_gt_std': num_extrema_gt_std
    }

    return output

def eval_collection__avg_num_kpts(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_names = sorted(ev.eval_config['set_names'])
    file_system = ev.file_system

    avgs = []
    weights = []
    for set_name in set_names:
        avgs.append(obj[collection_name][set_name]['avg_num_kpts'])
        weights.append(len(file_system[collection_name][set_name]))

    avgs = np.array(avgs).astype('float32')
    weights = np.array(weights).astype('float32')
    total = np.sum(weights)
    output = np.mean((avgs * weights) / total)

    return output

def eval_collection__std_num_kpts(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_names = sorted(ev.eval_config['set_names'])
    file_system = ev.file_system

    avgs = []
    weights = []
    for set_name in set_names:
        avgs.append(obj[collection_name][set_name]['avg_num_kpts'])
        weights.append(len(file_system[collection_name][set_name]))

    avgs = np.array(avgs).astype('float32')
    weights = np.array(weights).astype('float32')
    total = np.sum(weights)
    output = np.std((avgs * weights) / total)

    return output

def eval_collection__avg_num_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_names = sorted(ev.eval_config['set_names'])
    epsilon = ev.eval_config['epsilon']
    file_system = ev.file_system

    avgs = []
    weights = []
    for set_name in set_names:
        avgs.append(obj[collection_name][set_name]['avg_num_matching_kpts_for_e_{}'.format(epsilon)])
        weights.append(len(file_system[collection_name][set_name]))

    avgs = np.array(avgs).astype('float32')
    weights = np.array(weights).astype('float32')
    total = np.sum(weights)
    output = np.mean((avgs * weights) / total)

    return output

def eval_collection__std_num_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_names = sorted(ev.eval_config['set_names'])
    epsilon = ev.eval_config['epsilon']
    file_system = ev.file_system

    avgs = []
    weights = []
    for set_name in set_names:
        avgs.append(obj[collection_name][set_name]['avg_num_matching_kpts_for_e_{}'.format(epsilon)])
        weights.append(len(file_system[collection_name][set_name]))

    avgs = np.array(avgs).astype('float32')
    weights = np.array(weights).astype('float32')
    total = np.sum(weights)
    output = np.std((avgs * weights) / total)

    return output

def eval_collection__avg_perc_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_names = sorted(ev.eval_config['set_names'])
    epsilon = ev.eval_config['epsilon']
    file_system = ev.file_system

    avgs = []
    weights = []
    for set_name in set_names:
        avgs.append(obj[collection_name][set_name]['avg_perc_matching_kpts_for_e_{}'.format(epsilon)])
        weights.append(len(file_system[collection_name][set_name]))

    avgs = np.array(avgs).astype('float32')
    weights = np.array(weights).astype('float32')
    total = np.sum(weights)
    output = np.mean((avgs * weights) / total)

    return output


def eval_collection__std_perc_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> float:
    collection_name = ev.eval_config['collection_name']
    set_names = sorted(ev.eval_config['set_names'])
    epsilon = ev.eval_config['epsilon']
    file_system = ev.file_system

    avgs = []
    weights = []
    for set_name in set_names:
        avgs.append(obj[collection_name][set_name]['avg_perc_matching_kpts_for_e_{}'.format(epsilon)])
        weights.append(len(file_system[collection_name][set_name]))

    avgs = np.array(avgs).astype('float32')
    weights = np.array(weights).astype('float32')
    total = np.sum(weights)
    output = np.std((avgs * weights) / total)

    return output

def eval_collection__stats_num_kpts_for_e(ev:Evaluater, obj:Dict) -> Dict:
    collection_name = ev.eval_config['collection_name']
    set_names = sorted(ev.eval_config['set_names'])
    file_system = ev.file_system

    y = [obj[collection_name][set_name]['stats_num_kpts']['avg'] for set_name in set_names]
    y = np.array(y)

    val_min = np.min(y)
    val_max = np.max(y)
    idx_min = np.argmin(y)
    idx_max = np.argmax(y)

    avg = np.mean(y)
    std = np.std(y)

    lt_std = y < avg - std
    gt_std = y > avg + std
    condition = (lt_std) | (gt_std)
    val_extrema = y[condition].flatten()
    idx_extrema = np.where(condition)[0]
    num_extrema = len(val_extrema)

    val_extrema_lt_std = y[lt_std].flatten()
    idx_extrema_lt_std = np.where(lt_std)[0]
    num_extrema_lt_std = len(val_extrema_lt_std)

    val_extrema_gt_std = y[gt_std].flatten()
    idx_extrema_gt_std = np.where(gt_std)[0]
    num_extrema_gt_std = len(val_extrema_gt_std)

    output = {
        '_description': "Average number of found keypoints per set.",
        'set_names': set_names,
        'y': y,
        'avg': avg,
        'std': std,
        'val_min': val_min,
        'val_max': val_max,
        'idx_min': idx_min,
        'idx_max': idx_max,
        'val_extrema': val_extrema,
        'idx_extrema': idx_extrema,
        'num_extrema': num_extrema,
        'val_extrema_lt_std': val_extrema_lt_std,
        'idx_extrema_lt_std': idx_extrema_lt_std,
        'num_extrema_lt_std': num_extrema_lt_std,
        'val_extrema_gt_std': val_extrema_gt_std,
        'idx_extrema_gt_std': idx_extrema_gt_std,
        'num_extrema_gt_std': num_extrema_gt_std
    }

    return output

def eval_collection__stats_perc_matching_kpts_for_e(ev:Evaluater, obj:Dict) -> Dict:
    collection_name = ev.eval_config['collection_name']
    set_names = sorted(ev.eval_config['set_names'])
    file_system = ev.file_system
    epsilon = ev.eval_config['epsilon']

    y = [obj[collection_name][set_name]['stats_perc_matching_kpts_for_e_{}'.format(epsilon)]['avg'] for set_name in set_names]
    y = np.array(y)

    val_min = np.min(y)
    val_max = np.max(y)
    idx_min = np.argmin(y)
    idx_max = np.argmax(y)

    avg = np.mean(y)
    std = np.std(y)

    lt_std = y < avg - std
    gt_std = y > avg + std
    condition = (lt_std) | (gt_std)
    val_extrema = y[condition].flatten()
    idx_extrema = np.where(condition)[0]
    num_extrema = len(val_extrema)

    val_extrema_lt_std = y[lt_std].flatten()
    idx_extrema_lt_std = np.where(lt_std)[0]
    num_extrema_lt_std = len(val_extrema_lt_std)

    val_extrema_gt_std = y[gt_std].flatten()
    idx_extrema_gt_std = np.where(gt_std)[0]
    num_extrema_gt_std = len(val_extrema_gt_std)

    output = {
        '_description': "Average percentage of matching keypoints per set in whole collection",
        'set_names': set_names,
        'y': y,
        'avg': avg,
        'std': std,
        'val_min': val_min,
        'val_max': val_max,
        'idx_min': idx_min,
        'idx_max': idx_max,
        'val_extrema': val_extrema,
        'idx_extrema': idx_extrema,
        'num_extrema': num_extrema,
        'val_extrema_lt_std': val_extrema_lt_std,
        'idx_extrema_lt_std': idx_extrema_lt_std,
        'num_extrema_lt_std': num_extrema_lt_std,
        'val_extrema_gt_std': val_extrema_gt_std,
        'idx_extrema_gt_std': idx_extrema_gt_std,
        'num_extrema_gt_std': num_extrema_gt_std
    }

    return output

