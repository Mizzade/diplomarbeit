import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_names', nargs='+',
    help='Select the models to run the repeatability evaluation.',
    default=['sift', 'lift', 'superpoint', 'tilde'])

parser.add_argument('--max_size',type=str,
    help='Look for files with this max size postfix.',
    default='1300')

parser.add_argument('--allowed_extensions', nargs='+',
    help='Set the allowed file extensions for the imag directory.',
    default=['.png'])

parser.add_argument('--epsilons', nargs='+',
    help='Maximal pixel difference of keypoints to still be considered a match.',
    default=[1, 3, 7, 10])

parser.add_argument('--collection_names', nargs='+',
    help='Names of all collections to use.',
    default=['webcam'])

parser.add_argument('--set_names', nargs='+',
    help='Names of all sets to be used.',
    default=['chamonix', 'courbevoie', 'frankfurt', 'mexico', 'panorama', 'stlouis'])

parser.add_argument('--images_dir', type=str,
    help='Path to the image directory containing the collections.',
    default='../../data')

parser.add_argument('--data_dir', type=str,
    help='Path to the data dir containing the outputs of the detectors.',
    default='../../outputs')

parser.add_argument('--output_dir', type=str,
    help='Path to where to store this evaluation outputs.',
    default='outputs')

parser.add_argument('--output_file_prefix', type=str,
    help='Prefix for output file name.',
    default='repeatability_')

table_detector_names = {
    'sift': 'SIFT',
    'lift': 'LIFT',
    'superpoint': 'SuperPoint',
    'tilde': 'TILDE'
}

def get_config(argv):
    config, _ = parser.parse_known_args()
    config = vars(config)
    config['detector_names'] = [table_detector_names[x] for x in config['model_names']]
    config['kpts_file_format'] = 'kpts_{}__{}_{}.csv' \
        if config['max_size'] is None else 'kpts_{}__{}_{}_{}.csv'

    return config
