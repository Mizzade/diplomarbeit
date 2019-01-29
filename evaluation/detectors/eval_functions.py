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

def eval__num_kpts(ev:Evaluater) -> pd.DataFrame:
    """Returns pandas dataframe containing the number of found keypoints for
    each file in ev.file_list

    Arguments:
        ev {Evaluater} -- Evaluater object

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

