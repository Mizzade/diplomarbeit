from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import pickle
from collections import Counter
from Evaluater2 import Evaluater

def store_meta_data(
    data:Dict,
    config:Dict,
    fs:Dict,
    eval_config:Dict) -> Dict:
    out = {
        'config': config,
        'fs': fs
    }

    return out

def df_image_pair__recall_precision(
    data:Dict,
    config:Dict,
    fs:Dict,
    eval_config:Dict) -> pd.DataFrame:

    assert config['root_dir']
    assert config['data_dir']
    assert config['detector_name']

    assert eval_config['i'] is not None
    assert eval_config['j'] is not None
    assert eval_config['collection_name'] is not None
    assert eval_config['set_name'] is not None

    i = eval_config['i']
    j = eval_config['j']
    collection_name = eval_config['collection_name']
    set_name = eval_config['set_name']
    file_names = fs[collection_name][set_name]

    set_path = os.path.join(
        config['root_dir'],
        config['data_dir'],
        eval_config['collection_name'],
        eval_config['set_name'],
        'keypoints',
        config['detector_name'])

    kp_query = pd.read_csv(os.path.join(set_path, file_names[i]),
        sep=',', comment='#', usecols=[0, 1]).values
    kp_target = pd.read_csv(os.path.join(set_path, file_names[j]),
        sep=',', comment='#', usecols=[0, 1]).values

    T = 1.0 # epsilon in config. TODO
    t = np.linalg.norm(np.array([T, T]))

    # Compute norm for each 2d vector of kp_query with all 2d vectors of
    # kp_target
    norm = np.linalg.norm(kp_query[:, np.newaxis] - kp_target, axis=2)

    # For each kp in query find the index of the kp in target with
    # the smallest norm.
    min_idx_i = np.argmin(norm, axis=1)

    # Also find the min value for each row in kp_query
    min_val_i = np.min(norm, axis=1)

    # Build the labels (+1) for each kp in query.
    labels = np.ones_like(min_val_i)

    # Set all labels to (0) if the smallest norm is larger than threshold t
    labels[min_val_i > T] = 0

    # Find indices of kp_target, that are the best kps for kps_query multiple
    # times
    duplicate_indices = [item for item, count in Counter(min_idx_i).items() \
        if count > 1]

    # Find the positions in min_idx_i, that contain those duplicate indices
    # and set the corresponding labels to (0).
    pos_duplicate_indices = np.isin(min_idx_i, duplicate_indices)
    labels[pos_duplicate_indices] = 0

    iou_i = []
    for idx in range(kp_query.shape[0]):
        if labels[idx] == 0:
            iou_i.append(0)
        else:
            i = idx
            j = min_idx_i[idx]
            kpA = kp_query[i]
            kpB = kp_target[j]
            rA = _build_rect(kpA, 0.5*t)
            rB = _build_rect(kpB, 0.5*t)
            interArea = _area(rA, rB)
            iou = interArea / (2 * t * t - interArea)
            iou_i.append(iou)

    df = pd.DataFrame({
        'dist': min_val_i,
        'idx_target': min_idx_i.astype(np.int),
        'label': labels.astype(np.int),
        'iou_i': pd.Series(iou_i),
    })

    # sort df indices by norm, ascending
    sorted_indices = np.argsort(df['dist'].values)
    df = df.iloc[sorted_indices]

    # count labels from 1 to N
    index_count = np.arange(kp_query.shape[0]) + 1
    cum_sum_labels = np.cumsum(labels)

    # number of labels with (1)
    num_correct_labels = np.sum(labels)

    # precision for rank i
    prec_i = cum_sum_labels / index_count
    df.loc[:, 'precision_i'] = pd.Series(prec_i)

    # recall for rank i
    if num_correct_labels == 0:
        rec_i = np.zeros_like(cum_sum_labels, dtype=np.float)
    else:
        rec_i = np.divide(cum_sum_labels, num_correct_labels, out=np.zeros_like(cum_sum_labels, dtype=np.float), where=num_correct_labels==0)
    df.loc[:, 'recall_i'] = pd.Series(rec_i)

    return df

def _build_rect(p, r=0.5):
    return [p[0] - r, p[1] - r, p[0] + r, p[1] + r]

def _area(a, b):
  dx = min(a[2], b[2]) - max(a[0], b[0])
  dy = min(a[3], b[3]) - max(a[1], b[1])
  if (dx >= 0) and (dy >= 0):
    return dx * dy
  else:
    return 0.0

def float_image_pair__average_precision(
    data:Dict,
    config:Dict,
    fs:Dict,
    eval_config:Dict) -> float:

    i = eval_config['i']
    j = eval_config['j']
    collection_name = eval_config['collection_name']
    set_name = eval_config['set_name']

    df = data['collections'][collection_name]['sets'][set_name]\
        ['image_pairs']['{}_{}'.format(i, j)]

    try:
        true_positives = df.precision_i.values[df.label == 1]
        if len(true_positives) == 0:
            average_precision = 0
        else:
            average_precision = np.mean(true_positives)
    except:
        average_precision = 0

    return average_precision

def float_set__mean_average_precision(
    data:Dict,
    config:Dict,
    fs:Dict,
    eval_config:Dict) -> float:

    collection_name = eval_config['collection_name']
    set_name = eval_config['set_name']

    _d = data['collections'][collection_name]['sets'][set_name]['image_pairs']
    ap_values = np.array([_d[x] for x in _d if x.startswith('ap')])
    np.nan_to_num(ap_values, copy=False)

    return np.mean(ap_values)

def np_set__precision_recall_curve(
    data:Dict,
    config:Dict,
    fs:Dict,
    eval_config:Dict) -> np.array:

    collection_name = eval_config['collection_name']
    set_name = eval_config['set_name']
    _d = data['collections'][collection_name]['sets'][set_name]['image_pairs']
    img_pairs = [x for x in list(_d.keys()) if not x.startswith('ap')]

    xs = np.arange(0, 1.1, 0.1)
    ys = []

    for pair in img_pairs:
        df_i = _d[pair]
        ys_i = []
        for i in xs:
            try:
                maxP = np.max(df_i.precision_i.values[df_i.recall_i.values >= i])
            except:
                maxP = np.float(0)
            ys_i.append(maxP)
        ys.append(ys_i)

    ys = np.array(ys).mean(axis=0)

    # Returns x and y values, stacked verticall, so that row0 is xs, and
    # row1 is ys.
    return np.vstack([xs, ys])


def float_collection__mean_average_precision(
    data:Dict,
    config:Dict,
    fs:Dict,
    eval_config:Dict) -> float:

    collection_name = eval_config['collection_name']
    set_names = list(data['collections'][collection_name]['sets'].keys())

    _d = data['collections'][collection_name]['sets']
    map_values = np.array([_d[set_name]['map'] for set_name in set_names])
    np.nan_to_num(map_values, copy=False)

    return np.mean(map_values)

def np_collection__mean_precision_recall_curve(
    data:Dict,
    config:Dict,
    fs:Dict,
    eval_config:Dict) -> float:

    collection_name = eval_config['collection_name']
    set_names = list(data['collections'][collection_name]['sets'].keys())

    _d = data['collections'][collection_name]['sets']

    xs = np.arange(0, 1.1, 0.1)
    ys = np.mean(np.vstack([_d[set_name]['prcurve'][1] for set_name in set_names]), axis=0)

    return np.vstack([xs, ys])
