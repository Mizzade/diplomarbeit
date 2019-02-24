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

def eval_img_pair__recall_precision(
    collection_name:str,
    set_name:str,
    i:int,
    j:int) -> Dict:
    return {
        'key': ['collections', collection_name, 'sets', set_name, 'image_pairs', '{}_{}'.format(i, j)],
        'func': ef.df_image_pair__recall_precision,
        'collection_name': collection_name,
        'set_name': set_name,
        'i': i,
        'j': j
    }
