from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import pickle
import eval_support_functions as esf
from Evaluater2 import Evaluater
import eval_functions2 as ef
import eval_configs as ec

def fill_eval_queue(config:Dict, ev:Evaluater) -> Evaluater:
    if config['eval_meta__settings']:
        ev.add_eval_case(ec.eval_meta())

    for collection_name in list(ev.fs.keys()):
        for set_name in list(ev.fs[collection_name].keys()):

            file_names = list(ev.fs[collection_name][set_name])
            num_files = len(file_names)
            for i in range(num_files):
                for j in range(num_files):
                    if i != j:
                        if config['eval_imagepair__recall_precision']:
                            ev.add_eval_case(ec.eval_img_pair__recall_precision(
                                collection_name, set_name, i, j))

    return ev

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

    config = config_file[0]

    print('Start evaluation of detector {}.'.format(config['detector_name']))
    ev = esf.build_evaluater_for_detectors(config)
    ev = fill_eval_queue(config, ev)
    ev.run()
    print('Evaluation of detector {} done.'.format(config['detector_name']))

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
