from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import pickle
from Evaluater2 import Evaluater

def get_collection_names(config:Dict, sorted_output:bool=True) -> List[str]:
    """Returns the names of the collections specified in the config's
    `collection_names` parameter that are also inside the data directory.

    Arguments:
        config {Dict} -- See scripts/eval/config_eval_detectors.py

    Keyword Arguments:
        sorted_output {bool} -- Whether to sort the collection names.
            (default: {True})

    Returns:
        List[str] -- Intersection of collection names within `config` and
        `data_dir`.
    """
    assert config['root_dir']
    assert config['data_dir']

    data_dir = os.path.join(config['root_dir'], config['data_dir'])
    collection_names_in_data_dir = os.listdir(data_dir)
    collection_names = [name for name in collection_names_in_data_dir \
        if os.path.isdir(os.path.join(data_dir, name))]

    if config['collection_names'] is not None:
        collection_names = [name for name in collection_names \
            if name in config['collection_names']]

    if sorted_output:
        collection_names = sorted(collection_names)

    return collection_names


def get_set_names_for_collection(
    config:Dict,
    collection_name:str,
    sorted_output:bool=True) -> List[str]:
    """Finds all set names for a collection and returns thos set names, that
    are also specified within the `config` object. If `set_names` in `config` is
    `None`, return all set names.

    Arguments:
        config {Dict} -- See config_eval_detectors.py
        collection_name {str} -- Name of the collection to find all set names for.

    Keyword Arguments:
        sorted_output {bool} -- Whether to sort the set names. (default: {True})

    Returns:
        List[str] -- Intersection of set names found within the collection and
        set names specified within `config` object.
    """

    assert config['root_dir']
    assert config['data_dir']
    collection_path = os.path.join(
        config['root_dir'], config['data_dir'], collection_name)

    # Find all set folders.
    set_names = [set_name for set_name in os.listdir(collection_path) \
        if os.path.isdir(os.path.join(collection_path, set_name))]

    # Only take those set names listed in the config object. If `set_names`
    # is `None`, take all sets.
    if config['set_names']:
        set_names = [set_name for set_name in set_names \
            if set_name in config['set_names']]

    if sorted_output:
        set_names = sorted(set_names)

    return set_names


def get_keypoint_files_in_set_for_collection(
    config:Dict,
    collection_name:str,
    set_name:str,
    sorted_output:bool=True) -> List[str]:
    """Get all keypoint files for a collection and set that fit the
    `kpts_file_format` parameter within `config` object.

    Arguments:
        config {Dict} -- See config_eval_detectors.py
        collection_name {str} -- Name of a collection in `data_dir`.
        set_name {str} -- Name of a set within that colleciton.

    Keyword Arguments:
        sorted_output {bool} -- Whether to return a sorted list of the files
        within the set. (default: {True})

    Returns:
        List[str] -- All file names for that collection/set combination that fit
        `kpts_file_format` parameter filled with `max_num_keypoints` and
        `max_size` values of `config` object.
    """
    assert config['root_dir']
    assert config['data_dir']
    assert config['detector_name']
    assert config['kpts_file_format']
    set_path = os.path.join(
        config['root_dir'],
        config['data_dir'],
        collection_name,
        set_name,
        'keypoints',
        config['detector_name'])

    file_names = os.listdir(set_path)

    # Only get files within set path
    file_names = [fname for fname in file_names \
        if os.path.isfile(os.path.join(set_path, fname))]

    # Only get files that fit the parametes within config object
    file_names = [fname for fname in file_names \
        if config['kpts_file_format'] \
            .format('', config['max_num_keypoints'], config['max_size']) in fname]

    if sorted_output:
        file_names = sorted(file_names)

    return file_names


def build_file_system(
    config:Dict,
    sorted_output:bool=True) -> Dict:
    """Builds an dictionary, containing all collections names. Each collection
    name has a dictionary as value, whose keys are the corresponding set names
    of the collection. The values in turn are the file names of that set.

    Arguments:
        config {Dict} -- See config_eval_detectors.py

    Keyword Arguments:
        sorted_output {bool} -- Whether to sort all values (default: {True})

    Returns:
        Dict -- The filesystem
    """
    assert config['root_dir']
    assert config['data_dir']

    collection_names = get_collection_names(config, sorted_output=sorted_output)

    file_system = {}
    for collection_name in collection_names:
        file_system[collection_name] = {}

        set_names = get_set_names_for_collection(
            config, collection_name, sorted_output=sorted_output)

        for set_name in set_names:
            file_system[collection_name][set_name] = \
                get_keypoint_files_in_set_for_collection(
                    config, collection_name, set_name,sorted_output=sorted_output)

    return file_system

def build_evaluater_for_detectors(config:Dict) -> Evaluater:
    fs = build_file_system(config)
    return Evaluater('detectors', config, fs)

def create_dir(path: str) -> None:
    """Creates folder at given filepath `path`

    Arguments:
        path {str} -- Folderpath and subfolders to be crated.

    Returns:
        None
    """

    if path is not None and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

