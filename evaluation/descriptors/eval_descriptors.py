import numpy as np
import pandas as pd
import cv2
import os
import pickle
import sys
import collections
from itertools import combinations
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import config_eval_descriptors as cfg_desc
import eval_methods as em
import eval_support_functions as esf

"""Main File für die Evaluation von Deskriptoren."""

def get_metric_for_set(
    model_name:str,
    detector_name:str,
    descriptor_name:str,
    collection_name:str,
    set_name:str,
    config: Dict) -> Dict:
    print('Get metric for model {} and set {}.'.format(model_name, set_name))
    metrics = {}


    image_names = esf.get_image_names(collection_name, set_name, config)
    kpts_files = esf.get_keypoint_files(model_name, detector_name, 
        collection_name, set_name, image_names, config)
    desc_files = esf.get_descriptor_files(model_name, detector_name, descriptor_name,
        collection_name, set_name, image_names, config)

    image_pair_metrics = em.eval_image_pairs(kpts_files, desc_files, config)
    metrics = {**image_pair_metrics}

    return metrics


def get_metrics_for_collection(
    model_name:str,
    detector_name:str,
    descriptor_name:str,
    collection_name:str,
    config:Dict) -> Dict:
    metrics = {}
    for set_name in config['set_names']:
        metrics[set_name] = get_metric_for_set(
            model_name, 
            detector_name,
            descriptor_name,
            collection_name, 
            set_name, 
            config)
    return metrics

def get_metrics(
    model_name:str,
    detector_name:str,
    descriptor_name:str,
    config: Dict) -> Dict:
    metrics = {}
    for collection_name in config['collection_names']:
        metrics[collection_name] = get_metrics_for_collection(
            model_name,
            detector_name, 
            descriptor_name, 
            collection_name, 
            config)
    return metrics

def main(config: Dict) -> None:
    for model_name, detector_name, descriptor_name in tqdm(list(zip(config['model_names'], config['detector_names'], config['descriptor_names']))):
        output_name = 'desc_{}_with_det_{}.pkl'.format(descriptor_name, detector_name)
        dst_name = ''.join([config['output_file_prefix'], output_name])
        dst_file_path = os.path.join(config['output_dir'], dst_name)
        metrics = get_metrics(model_name, detector_name, descriptor_name, config)
        esf.save_metrics(metrics, dst_file_path)

if __name__ == '__main__':
    argv = sys.argv[1:]
    config = cfg_desc.get_config(argv)
    if config['dry']:
        print(config)
    else:
        main(config)