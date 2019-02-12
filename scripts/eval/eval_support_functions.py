import os
from typing import List, Dict, Tuple, Any
import pickle
import shutil

def get_set_names(config:Dict, sorted_output:bool=True) -> List[str]:
    """Given a list on collection within the config object, return all set names
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

def get_keypoint_file_list_for_collection(
    collection_name:str,
    config:Dict,
    sorted_output=True) -> List[str]:

    data_dir = config['data_dir']
    set_names = config['set_names']
    detector_name = config['detector_name']
    max_size = '' if config['max_size'] is None else str(config['max_size'])
    allowed_extensions = config['allowed_extensions']


    file_list = []
    collection_path = os.path.join(data_dir, collection_name)

    for set_name in next(os.walk(collection_path))[1]:
        if set_name in set_names or len(set_names) == 0:
            set_path = os.path.join(collection_path, set_name, 'keypoints', detector_name)

            for file in os.listdir(set_path):
                f_name, f_ext = os.path.splitext(file)
                if f_ext.lower() in allowed_extensions and max_size in f_name:
                    file_list.append(os.path.join(set_path, file))

    if sorted_output:
        file_list = sorted(file_list)

    return file_list

def get_file_list_for_keypoints(config:Dict, sorted_output=True) -> List[str]:
    file_list = []
    for collection_name in config['collection_names']:
        file_list = [] + get_keypoint_file_list_for_collection(collection_name, config, sorted_output)

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


def build_file_system(config:Dict, fs_type:str, sorted_output:bool=True):

    assert (fs_type == 'keypoints') or (fs_type == 'descriptors')

    output = {}

    for collection_name in config['collection_names']:
        output[collection_name] = {}
        output[collection_name]['_file_list'] = [] # Contains all files in collections
        collection_path = os.path.join(config['image_dir'], collection_name)

        for set_name in next(os.walk(collection_path))[1]:
            if set_name in config['set_names']:
                output[collection_name][set_name] = []
                set_path = os.path.join(collection_path, set_name)

                for file in os.listdir(set_path):
                    file_name, _ = os.path.splitext(file)

                    if fs_type == 'keypoints':
                        kpts_path = os.path.join(
                            config['data_dir'],
                            collection_name,
                            set_name,
                            fs_type,
                            config['detector_name'],
                            config['kpts_file_format'].format(file_name, config['max_size']))
                    elif fs_type == 'descriptors':
                        kpts_path = os.path.join(
                            config['data_dir'],
                            collection_name,
                            set_name,
                            fs_type,
                            config['descriptor_name'],
                            config['detector_name'],
                            config['kpts_file_format'].format(file_name, config['max_size']))

                    if os.path.exists(kpts_path):
                        output[collection_name][set_name].append(kpts_path)     # append file path to set of collection
                        output[collection_name]['_file_list'].append(kpts_path) # append file path to general list of files for collection

                # Remove entries with empty lists
                if len(output[collection_name][set_name]) == 0:
                    del output[collection_name][set_name]

    # Sort file lists for sets
    if sorted_output:
        for collection_name in config['collection_names']:
            for set_name in output[collection_name]:
                output[collection_name][set_name] = sorted(output[collection_name][set_name])

    return output




