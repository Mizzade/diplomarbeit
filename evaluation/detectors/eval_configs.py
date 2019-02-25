from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import pickle
import eval_support_functions as esf
import eval_functions2 as ef

def eval_meta() -> Dict:
    return {
        'key': ['meta'],
        'func': ef.store_meta_data
    }

def eval_image_pair__stats(
    collection_name:str,
    set_name:str,
    i:int,
    j:int) -> Dict:
    return {
        'key': ['collections', collection_name, 'sets', set_name, 'image_pairs', 'stats_{}_{}'.format(i, j)],
        'func': ef.dict_image_pair__stats,
        'collection_name': collection_name,
        'set_name': set_name,
        'i': i,
        'j': j
    }

def eval_image_pair__recall_precision(
    collection_name:str,
    set_name:str,
    i:int,
    j:int) -> Dict:
    return {
        'key': ['collections', collection_name, 'sets', set_name, 'image_pairs', 'df_{}_{}'.format(i, j)],
        'func': ef.df_image_pair__recall_precision,
        'collection_name': collection_name,
        'set_name': set_name,
        'i': i,
        'j': j
    }

def eval_image_pair__average_precision(
    collection_name:str,
    set_name:str,
    i:int,
    j:int) -> Dict:
    return {
        'key': ['collections', collection_name, 'sets', set_name, 'image_pairs', 'ap_{}_{}'.format(i, j)],
        'func': ef.float_image_pair__average_precision,
        'collection_name': collection_name,
        'set_name': set_name,
        'i': i,
        'j': j
    }

def eval_set__stats(
    collection_name:str,
    set_name:str) -> Dict:
    return {
        'key': ['collections', collection_name, 'sets', set_name, 'stats'],
        'func': ef.dict_set__stats,
        'collection_name': collection_name,
        'set_name': set_name
    }

def eval_set__mean_average_precision(
    collection_name:str,
    set_name:str) -> Dict:
    return {
        'key': ['collections', collection_name, 'sets', set_name, 'map'],
        'func': ef.float_set__mean_average_precision,
        'collection_name': collection_name,
        'set_name': set_name
    }

def eval_set__precision_recall_curve(
    collection_name:str,
    set_name:str) -> Dict:
    return {
        'key': ['collections', collection_name, 'sets', set_name, 'prcurve'],
        'func': ef.np_set__precision_recall_curve,
        'collection_name': collection_name,
        'set_name': set_name
    }

def eval_collection__stats(
    collection_name:str) -> Dict:
    return {
        'key': ['collections', collection_name, 'stats'],
        'func': ef.dict_collection__stats,
        'collection_name': collection_name
    }

def eval_collection__mean_average_precision(
    collection_name:str) -> Dict:
    return {
        'key': ['collections', collection_name, 'map'],
        'func': ef.float_collection__mean_average_precision,
        'collection_name': collection_name
    }

def eval_collection__mean_precision_recall_curve(
    collection_name:str) -> Dict:
    return {
        'key': ['collections', collection_name, 'mprcurve'],
        'func': ef.np_collection__mean_precision_recall_curve,
        'collection_name': collection_name
    }
