from typing import List, Tuple, Dict, Any, Callable
import cv2
import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import pickle
import eval_support_functions2 as esf

class EvalCase():
    def __init__(self, config:Dict) -> None:
        self.config = config

    def run(self, data:Dict, config:Dict, fs:Dict) -> Tuple[str, Any]:
        """Runs a evaluation case. Returns the keypath and the value of the
        evaluation.

        Arguments:
            data {Dict} -- Data object in which to store the this evaluation's
            value. Might be used to get already computed value from.
            config {Dict} -- Config ojbect. See config_eval_descriptors.py
            fs {Dict} -- Filesystem, with collection names, corresponding set
            names and file names.

        Returns:
            Tuple[str, Any] -- keypath where to store value in DATA.
        """

        key = self.config['key']
        value = self.config['func'](data, config, fs, self.config)

        return key, value

class Evaluater():
    def __init__(self,
        etype:str, # detectors | descriptors
        config:Dict,
        fs:Dict) -> None:
        self.config = config
        self.fs = fs
        self.eval_cases = []

        # Where to save the results.
        self.output_path = os.path.join(
            config['root_dir'],
            config['output_dir'],
            etype,
            config['eval_file_format'].format(
                config['detector_name'],
                config['max_num_keypoints'],
                config['max_size'])
        )

        # Example:
        # root/output_evaluation/descriptors/desc_X_det_Y_1000_1200.pkl

    def add_eval_case(self, test_case_config:Dict) -> None:
        self.eval_cases.append(EvalCase(test_case_config))

    def load_data(self) -> Dict:
        try:
            with open(self.output_path, 'rb') as src:
                data = pickle.load(src, encoding='utf-8')
        except FileNotFoundError:
            data = {}

        return data

    def save_data(self, data:Dict) -> None:
        head, _ = os.path.split(self.output_path)
        esf.create_dir(head)

        try:
            with open(self.output_path, 'wb') as dst:
                pickle.dump(data, dst, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Something went wrong while saving.\n ', e)

    def insert_data(
        self,
        target:Dict,
        keypath:List[str],
        value:Any) -> Dict:
        if type(target) is not dict:
            raise AttributeError('Evaluater.insert_data() exspects dict as first argument')
        if len(keypath) == 0:
            raise AttributeError('Evaluater.insert_data() espects a list of keys with at least one key in it. O were given.')

        _target = target
        for key in keypath[:-1]:
            try:
                _target =_target[key]
            except KeyError:
                    _target[key] = {}
                    _target = _target[key]

        _target[keypath[-1]] = value

        return target

    def run(self) -> None:
        data = self.load_data()

        for test_case in tqdm(self.eval_cases):
            key, value = test_case.run(data, self.config, self.fs)
            data = self.insert_data(data, key, value)
            self.save_data(data)

