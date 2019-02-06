import os
from typing import List, Dict, Tuple, Any
import pickle
import shutil

def get_collection_names(config:Dict, sorted_output:bool=True) -> List[str]:
    """Given the config object, return all collection names set by the user.
    if no collection name is set, use DATA_DIR to find all collections in the
    the data (images).
    """


    if config['collection_names'] is not None:
        # Get the collection names from config object.
        collection_names = config['collection_names']

    else:
        # Find all collections in data dir.
        path_to_data_dir = os.path.join(config['root_dir'], config['data_dir'])
        collection_names = [d for d in os.listdir(path_to_data_dir) \
            if os.path.isdir(os.path.join(path_to_data_dir, d))]

    if sorted_output:
        collection_names = sorted(collection_names)

    return collection_names

def get_set_names(config:Dict, sorted_output:bool=True) -> List[str]:
    """Given a list of collections within the config object, return all set names
    set by the user. If no set name was given, find all set names
    for all collections and return them as a list of strings.
    """

    if config['set_names'] is not None:
        # Get the set names from config object.
        set_names = config['set_names']

    else:
        # Find all sets within all collections and return unique list.
        set_names = []
        for c in config['collection_names']:
            path_to_collection_dir = os.path.join(config['root_dir'], config['data_dir'], c)
            set_names +=  [d for d in os.listdir(path_to_collection_dir) \
                if os.path.isdir(os.path.join(path_to_collection_dir, d))]
        set_names = list(set(set_names))
    if sorted_output:
        set_names = sorted(set_names)

    return set_names

def get_file_list(config:Dict, sorted_output=True) -> List[str]:
    """Given the config object, return all absolute paths to the files in the
    selected collections and sets."""

    data_dir = config['data_dir']
    allowed_extensions = config['allowed_extensions']
    collection_names = config['collection_names']
    set_names = config['set_names']
    max_num_images = config['max_num_images']
    skip_first_n = config['skip_first_n']


    file_list = []
    for collection_name in next(os.walk(data_dir))[1]:
        if collection_name in collection_names or len(collection_names) == 0:
            collection_path = os.path.join(data_dir, collection_name)

            for set_name in next(os.walk(collection_path))[1]:
                if set_name in set_names or len(set_names) == 0:
                    set_path = os.path.join(collection_path, set_name)

                    for file in os.listdir(set_path):
                        _, f_ext = os.path.splitext(file)
                        if f_ext.lower() in allowed_extensions:
                            file_list.append(os.path.join(set_path, file))

    if sorted_output:
        file_list = sorted(file_list)

    # Skip first n
    file_list = file_list[skip_first_n:]

    # Take only m images
    if max_num_images:
        file_list = file_list[:max_num_images]
    return file_list

def print_configuration(config:Dict) -> None:
    print('\n{}\n--------'.format(config['configuration_name']))
    for k,v in config.items():
        print('{}: {}'.format(k, v))
    print('------\n')

def create_dir(path: str):
    if path is not None and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def remove_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def write_config_file(path_to_config_file:str, data_list: List[Any]) -> None:

   with open(path_to_config_file, 'wb') as dst:
        pickle.dump(
            data_list,
            dst,
            protocol=pickle.HIGHEST_PROTOCOL)






