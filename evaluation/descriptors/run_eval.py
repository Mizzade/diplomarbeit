from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import json
import os
from tqdm import tqdm
import pickle
# import eval_functions as efunc
# from Evaluater import Evaluater

def main(argv: Tuple[str]) -> None:
    """Runs evaluation on a detector saves the results.

    Arguments:
        argv {Tuple[str]} -- List of one parameter. There should be exactly
            one parameter - the path to the config file inside the tmp dir.
            This config file will be used to get all other information and
    """
    if len(argv) <= 0:
        raise RuntimeError("Missing argument <path_to_config_file>. Abort")

    with open(argv[0], 'rb') as src:
        config_file = pickle.load(src, encoding='utf-8')

    config, file_system = config_file

    print('Start evaluation of descriptor {} with detector {}.'.format(config['descriptor_name'], config['detector_name']))

    #list_of_evaluations = build_list_of_evaluations(config, file_system)
    #run_evaluations(list_of_evaluations)

    print('Evaluation of  evaluation of descriptor {} with detector {} done.'.format(config['descriptor_name'], config['detector_name']))

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
