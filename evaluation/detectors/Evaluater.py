from typing import List, Tuple, Dict, Any, Callable
import cv2
import numpy as np
import pandas as pd
import sys
import json
import os
from tqdm import tqdm
import pickle

class Evaluater():

    def __init__(self,
        target_keypath:List[str],
        config:Dict,
        file_system: Dict,
        eval_func:Callable,
        eval_config:Dict=None,
        requirements:Dict=None) -> None:

        self.config = config
        self.file_system = file_system
        self.eval_func = eval_func
        self.eval_config = eval_config
        self.requirements = requirements
        self.target_keypath = target_keypath # Where to save the data in target dict.

        # Where to save the data
        self.output_path = os.path.join(config['output_dir'], 'detectors', config['detector_name'] + '.pkl')

    def run(self):
        obj = self._load_data()
        data = self.eval_func(self, obj)
        self._insert_key_and_value(obj, data)
        self._save_data(obj)

    def _load_data(self) -> Dict:
        try:
            with open(self.output_path, 'rb') as src:
                data = pickle.load(src, encoding='utf-8')
        except FileNotFoundError:
            data = {}

        return data

    def _save_data(self, data:Any) -> None:
        try:
            with open(self.output_path, 'wb') as dst:
                pickle.dump(data, dst, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Something went wrong while saving.\n ', e)

    # def _key_exists(self, obj:Dict, list_of_keys:List[List[str]]) -> bool:
    #     if type(obj) is not dict:
    #         raise AttributeError('_key_exists() exspects dict as first argument')
    #     if len(list_of_keys) == 0:
    #         raise AttributeError('_key_exists() espects a list of list of keys with at least one key in it. O were given.')

    #     for lok in list_of_keys:
    #         _obj = obj
    #         for key in lok:
    #             try:
    #                 _obj = _obj[key]
    #             except KeyError:
    #                 return False

    #     return True

    def _insert_key_and_value(self, target:Dict, value:Any) -> Dict:
        if type(target) is not dict:
            raise AttributeError('_insert_key() exspects dict as first argument')
        if len(self.target_keypath) == 0:
            raise AttributeError('_insert_key() espects a list of keys with at least one key in it. O were given.')

        _target = target
        for key in self.target_keypath[:-1]:
            try:
                _target =_target[key]
            except KeyError:
                    _target[key] = {}
                    _target = _target[key]

        _target[self.target_keypath[-1]] = value

        return target




