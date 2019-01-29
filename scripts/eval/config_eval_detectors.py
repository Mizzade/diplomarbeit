import argparse
import os
import eval_support_functions as esf

parser = argparse.ArgumentParser()

parser.add_argument('--configuration_name',
    type=str,
    help='Name of this configiration. This will be the title if DRY is set.',
    default='Configuration: eval_detectors.py')

parser.add_argument('--root_dir',
    type=str,
    help='Set the path to the root directory of this repository.',
    required=True)

parser.add_argument('--output_dir',
    type=str,
    help='Path to directory containing the results of the detectors. ' +
    'Relative to ROOT_DIR. Default: outputs',
    default='output_evaluation')

parser.add_argument('--data_dir',
    type=str,
    help='Path to the directory containing data sets from the detectors and ' +
    ' descriptors. Relative to ROOT_DIR. Default: outputs',
    default='outputs')

parser.add_argument('--image_dir',
    type=str,
    help='Path to the directory containing the image collections. Relative ' +
    'to ROO_DIR. Default: data',
    default='data')

parser.add_argument('--root_dir_detector',
    type=str,
    help='Path to the folder containing the functions for evaluating the ' +
    'detectors. Relative from ROOT_DIR. Default: evaluation/detectors',
    default=os.path.join('evaluation', 'detectors'))

parser.add_argument('--tmp_dir_detector',
    type=str,
    help='Path to temporary directory to save intermediate results and config ' +
    'and config file. Relative from ROOT_DIR_DETECTOR. Default: tmp',
    default='tmp')

parser.add_argument('--max_size',
    type=int,
    help='If descriptors and detectors have been run with MAX_SIXE paramater ' +
    'the resulting .csv files with have the corresponding postfix <_MAX_SIXE>. ' +
    'This allows to only select those files with the appropriate postfix. ' +
    'Default: None',
    default=None)

parser.add_argument('--detectors',
    nargs='+',
    help='Choose which detectors should be run. Default: (sift, tilde, lift, superpoint)',
    default=['sift', 'tilde', 'lift', 'superpoint'])

parser.add_argument('--collection_names',
    nargs='+',
    help='Name of all collections to be processed within the data_dir. ' +
    'Skip collections that are not found. A value of None means all ' +
    'collections. Default: None',
    default=['webcam'])

parser.add_argument('--set_names',
    nargs='+',
    help='Name of all sets to be processed within the data_dir. ' +
    'Skip sets that are not found. A value of None means all sets. ' +
    'Default: None',
    default=None)

parser.add_argument('--allowed_extensions',
    nargs='+',
    help='Set the allowed file extensions for the data directory. Only files ' +
    'with the fitting extension will be used by the detector. Default: ' +
    '(.csv)',
    default=['.csv'])

parser.add_argument('--dry',
    dest='dry',
    action='store_true',
    help='If set, only print config, but do not run models. Default: False',
    default=False)


def get_config(argv):
    config, _ = parser.parse_known_args()
    config = vars(config)

    config['set_names'] = esf.get_set_names(config)

    # Set absolute paths
    config['data_dir'] = os.path.join(config['root_dir'], config['data_dir'])
    config['image_dir'] = os.path.join(config['root_dir'], config['image_dir'])
    config['output_dir'] = os.path.join(config['root_dir'], config['output_dir'])
    config['root_dir_detector'] = os.path.join(config['root_dir'], config['root_dir_detector'])
    config['tmp_dir_detector'] = os.path.join(config['root_dir_detector'], config['tmp_dir_detector'])

    config['kpts_file_format'] = '{}.csv' if config['max_size'] is None else '{}_{}.csv'

    return config
