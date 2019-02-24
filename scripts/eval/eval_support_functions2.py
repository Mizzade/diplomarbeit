import os
from typing import List, Dict, Tuple, Any
import pickle
import shutil

def print_configuration(config:Dict) -> None:
    """Prints a given configuration dictionary, i.e. when --dry parameter has
    been set.

    Arguments:
        config {Dict} -- Configuration dictionary. See config_eval_detectors.py
            or config_eval_descriptors.py.

    Returns:
        None
    """

    assert config['configuration_name'] is not None

    print('\n{}\n--------'.format(config['configuration_name']))
    for k,v in config.items():
        print('{}: {}'.format(k, v))
    print('------\n')

def create_dir(path: str) -> None:
    """Creates folder at given filepath `path`

    Arguments:
        path {str} -- Folderpath and subfolders to be crated.

    Returns:
        None
    """

    if path is not None and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def remove_dir(path: str) -> None:
    """Removes folder at given file path `path`.

    Arguments:
        path {str} -- Path to the folder to be removed.

    Returns:
        None
    """

    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def write_config_file(path_to_config_file:str, data_list: List[Any]) -> None:
    """Writes a pickle file to `path_to_config_file` containing `data_list`.

    Arguments:
        path_to_config_file {str} -- Where to create the config file.
        data_list {List[Any]} -- Data to be stored inside pickle file.

    Returns:
        None
    """

    with open(path_to_config_file, 'wb') as dst:
        pickle.dump(
            data_list,
            dst,
            protocol=pickle.HIGHEST_PROTOCOL)
