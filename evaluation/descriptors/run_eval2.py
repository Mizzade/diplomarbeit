from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import pickle
import eval_support_functions2 as esf
from Evaluater2 import Evaluater
import eval_functions2 as ef
import eval_configs2 as ec

def fill_eval_queue(config:Dict, ev:Evaluater) -> Evaluater:
    """Given a config object and en Evaluater, fill the Evaluater's task queue
    with tasks depending on the config's settings. Returns the input Evaluater

    Arguments:
        config {Dict} -- See config_eval_descriptors.py
        ev {Evaluater} -- See descriptors/Evaluater2.py

    Returns:
        Evaluater -- The input Evaluater, now with modified internal state.
    """

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

    print('Start evaluation of descriptor {} wit hdetector {}.'.format(config['descriptor_name'], config['detector_name']))
    ev = esf.build_evaluater_for_detectors(config)
    #ev = fill_eval_queue(config, ev)
    #ev.run()
    fs = esf.build_file_system(config)
    print('fileSys:\n', fs)
    print('Evaluation of descriptor {} with detector {} done.'.format(config['descriptor_name'], config['detector_name']))

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
