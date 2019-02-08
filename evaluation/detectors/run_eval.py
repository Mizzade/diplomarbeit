from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import pandas as pd
import sys
import json
import os
from tqdm import tqdm
import pickle
import eval_functions as efunc
from Evaluater import Evaluater

def build_list_of_evaluations(config:Dict, file_system:Dict) -> None:
    list_of_evaluations = []

    if config['eval_meta__settings']:
        list_of_evaluations.append(Evaluater(
            ['_meta'],
            config,
            file_system,
            efunc.store_meta_information
        ))

    for collection_name in config['collection_names']:
        set_names = [s for s in file_system[collection_name] if not s.startswith('_')]

        # Find number of found keypoints for each file in collection.
        if config['eval_image__num_kpts']:
            list_of_evaluations.append(Evaluater(
                [collection_name, 'num_kpts'],
                config,
                file_system,
                efunc.eval_image__num_kpts,
                eval_config={
                    'collection_name': collection_name
                }))

        # Find number of maximal possible equal keypoints
        if config['eval_imagepair__max_num_matching_kpts']:
            for set_name in set_names:
                list_of_evaluations.append(Evaluater(
                    [collection_name, set_name, 'max_num_matching_kpts'],
                    config,
                    file_system,
                    efunc.eval_imagepair__num_max_matching_kpts,
                    eval_config={
                        'collection_name': collection_name,
                        'set_name': set_name
                    }))

        # Find number of matching kpts for all image pairs in each set.
        if config['eval_imagepair__num_matching_kpts']:
            for set_name in set_names:
                for epsilon in config['epsilons']:
                    for i in file_system[collection_name][set_name]:
                        for j in file_system[collection_name][set_name]:
                            list_of_evaluations.append(Evaluater(
                                [collection_name, set_name, 'num_matching_kpts_for_e_{}'.format(epsilon)],
                                config,
                                file_system,
                                efunc.eval__num_matching_kpts_for_e,
                                eval_config={
                                    'collection_name': collection_name,
                                    'set_name': set_name,
                                    'epsilon': epsilon,
                                    'i': i,
                                    'j': j
                                }
                            ))

        # Find the percentage of matchting kpts for all image pairs
        if config['eval_imagepair__perc_matching_kpts']:
            for set_name in set_names:
                for epsilon in config['epsilons']:
                    list_of_evaluations.append(Evaluater(
                        [collection_name, set_name, 'perc_matching_kpts_for_e_{}'.format(epsilon)],
                        config,
                        file_system,
                        efunc.eval_imagepair__perc_matching_keypoints_for_e,
                        eval_config={
                            'collection_name': collection_name,
                            'set_name': set_name,
                            'epsilon': epsilon,
                        }
                    ))

        # Average number of keypoints of all images in set.
        if config['eval_set__avg_num_kpts']:
            for set_name in set_names:
                list_of_evaluations.append(Evaluater(
                    [collection_name, set_name, 'avg_num_kpts'],
                    config,
                    file_system,
                    efunc.eval_set__avg_num_kpts,
                    eval_config={
                        'collection_name': collection_name,
                        'set_name': set_name
                    }
                ))

        if config['eval_set__std_num_kpts']:
            for set_name in set_names:
                list_of_evaluations.append(Evaluater(
                    [collection_name, set_name, 'std_num_kpts'],
                    config,
                    file_system,
                    efunc.eval_set__std_num_kpts,
                    eval_config={
                        'collection_name': collection_name,
                        'set_name': set_name
                    }
                ))
        if config['eval_set__stats_num_kpts']:
            for set_name in set_names:
                list_of_evaluations.append(Evaluater(
                    [collection_name, set_name, 'stats_num_kpts'],
                    config,
                    file_system,
                    efunc.eval_set__stats_num_kpts,
                    eval_config={
                        'collection_name': collection_name,
                        'set_name': set_name
                    }
                ))

        if config['eval_set__avg_num_matching_kpts']:
            for set_name in set_names:
                for epsilon in config['epsilons']:
                    list_of_evaluations.append(Evaluater(
                        [collection_name, set_name, 'avg_num_matching_kpts_for_e_{}'.format(epsilon)],
                        config,
                        file_system,
                        efunc.eval_set__avg_num_matching_kpts_for_e,
                        eval_config={
                            'collection_name': collection_name,
                            'set_name': set_name,
                            'epsilon': epsilon
                        }
                    ))

        if config['eval_set__std_num_matching_kpts']:
            for set_name in set_names:
                for epsilon in config['epsilons']:
                    list_of_evaluations.append(Evaluater(
                        [collection_name, set_name, 'std_num_matching_kpts_for_e_{}'.format(epsilon)],
                        config,
                        file_system,
                        efunc.eval_set__std_num_matching_kpts_for_e,
                        eval_config={
                            'collection_name': collection_name,
                            'set_name': set_name,
                            'epsilon': epsilon
                        }
                    ))


        if config['eval_set__avg_max_num_matching_kpts']:
            for set_name in set_names:
                list_of_evaluations.append(Evaluater(
                    [collection_name, set_name, 'avg_max_num_matching_kpts'],
                    config,
                    file_system,
                    efunc.eval_set__avg_max_num_matching_kpts,
                    eval_config={
                        'collection_name': collection_name,
                        'set_name': set_name
                    }
                ))

        if config['eval_set__std_max_num_matching_kpts']:
            for set_name in set_names:
                list_of_evaluations.append(Evaluater(
                    [collection_name, set_name, 'std_max_num_matching_kpts'],
                    config,
                    file_system,
                    efunc.eval_set__std_max_num_matching_kpts,
                    eval_config={
                        'collection_name': collection_name,
                        'set_name': set_name
                    }
                ))

        if config['eval_set__avg_perc_matchting_kpts']:
            for set_name in set_names:
                for epsilon in config['epsilons']:
                    list_of_evaluations.append(Evaluater(
                        [collection_name, set_name, 'avg_perc_matching_kpts_for_e_{}'.format(epsilon)],
                        config,
                        file_system,
                        efunc.eval_set__avg_perc_matchting_kpts_for_e,
                        eval_config={
                            'collection_name': collection_name,
                            'set_name': set_name,
                            'epsilon': epsilon
                        }
                    ))

        if config['eval_set__std_perc_matchting_kpts']:
            for set_name in set_names:
                for epsilon in config['epsilons']:
                    list_of_evaluations.append(Evaluater(
                        [collection_name, set_name, 'std_perc_matching_kpts_for_e_{}'.format(epsilon)],
                        config,
                        file_system,
                        efunc.eval_set__std_perc_matchting_kpts_for_e,
                        eval_config={
                            'collection_name': collection_name,
                            'set_name': set_name,
                            'epsilon': epsilon
                        }
                    ))

        if config['eval_set__stats_perc_matching_kpts']:
            for set_name in set_names:
                for epsilon in config['epsilons']:
                    list_of_evaluations.append(Evaluater(
                        [collection_name, set_name, 'stats_perc_matching_kpts_for_e_{}'.format(epsilon)],
                        config,
                        file_system,
                        efunc.eval_set__stats_perc_matching_kpts_for_e,
                        eval_config={
                            'collection_name': collection_name,
                            'set_name': set_name,
                            'epsilon': epsilon
                        }
                    ))

        if config['eval_collection__avg_num_kpts']:
            list_of_evaluations.append(Evaluater(
                [collection_name, 'avg_num_kpts'],
                config,
                file_system,
                efunc.eval_collection__avg_num_kpts,
                eval_config={
                    'collection_name': collection_name,
                    'set_names': set_names
                }
            ))

        if config['eval_collection__std_num_kpts']:
            list_of_evaluations.append(Evaluater(
                [collection_name, 'std_num_kpts'],
                config,
                file_system,
                efunc.eval_collection__std_num_kpts,
                eval_config={
                    'collection_name': collection_name,
                    'set_names': set_names
                }
            ))

        if config['eval_collection__avg_num_matching_kpts']:
            for epsilon in config['epsilons']:
                list_of_evaluations.append(Evaluater(
                    [collection_name, 'avg_num_matching_kpts_for_e_{}'.format(epsilon)],
                    config,
                    file_system,
                    efunc.eval_collection__avg_num_matching_kpts_for_e,
                    eval_config={
                        'collection_name': collection_name,
                        'set_names': set_names,
                        'epsilon': epsilon
                    }
                ))

        if config['eval_collection__std_num_matching_kpts']:
             for epsilon in config['epsilons']:
                list_of_evaluations.append(Evaluater(
                    [collection_name, 'std_num_matching_kpts_for_e_{}'.format(epsilon)],
                    config,
                    file_system,
                    efunc.eval_collection__std_num_matching_kpts_for_e,
                    eval_config={
                        'collection_name': collection_name,
                        'set_names': set_names,
                        'epsilon': epsilon
                    }
                ))

        if config['eval_collection__avg_perc_matching_kpts']:
            for epsilon in config['epsilons']:
                list_of_evaluations.append(Evaluater(
                    [collection_name, 'avg_perc_matching_kpts_for_e_{}'.format(epsilon)],
                    config,
                    file_system,
                    efunc.eval_collection__avg_perc_matching_kpts_for_e,
                    eval_config={
                        'collection_name': collection_name,
                        'set_names': set_names,
                        'epsilon': epsilon
                    }
                ))

        if config['eval_collection__std_perc_matching_kpts']:
            for epsilon in config['epsilons']:
                list_of_evaluations.append(Evaluater(
                    [collection_name, 'std_perc_matching_kpts_for_e_{}'.format(epsilon)],
                    config,
                    file_system,
                    efunc.eval_collection__std_perc_matching_kpts_for_e,
                    eval_config={
                        'collection_name': collection_name,
                        'set_names': set_names,
                        'epsilon': epsilon
                    }
                ))

        if config['eval_collection__stats_num_kpts']:
            for epsilon in config['epsilons']:
                list_of_evaluations.append(Evaluater(
                    [collection_name, 'stats_num_kpts_for_e_{}'.format(epsilon)],
                    config,
                    file_system,
                    efunc.eval_collection__stats_num_kpts_for_e,
                    eval_config={
                        'collection_name': collection_name,
                        'set_names': set_names,
                        'epsilon': epsilon
                    }
                ))

        if config['eval_collection__stats_perc_matching_kpts']:
            for epsilon in config['epsilons']:
                list_of_evaluations.append(Evaluater(
                    [collection_name, 'stats_perc_matching_kpts_for_e_{}'.format(epsilon)],
                    config,
                    file_system,
                    efunc.eval_collection__stats_perc_matching_kpts_for_e,
                    eval_config={
                        'collection_name': collection_name,
                        'set_names': set_names,
                        'epsilon': epsilon
                    }
                ))

    return list_of_evaluations


def run_evaluations(list_of_evaluations:List[Evaluater]) -> None:
    for e in tqdm(list_of_evaluations):
        e.run()


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

    print('Start evaluation of detector {}.'.format(config['detector_name']))

    list_of_evaluations = build_list_of_evaluations(config, file_system)
    run_evaluations(list_of_evaluations)

    print('Evaluation of detector {} done.'.format(config['detector_name']))

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
