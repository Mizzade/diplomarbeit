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

# Allow setting tests ON and OFF.

# Number of keypoints per image in each set.
parser.add_argument('--eval_set__num_kpts_per_image',
    dest='eval_set__num_kpts_per_image',
    action='store_true',
    help='Add number of keypoints per image for each set. Default: True.',
    default=None)

parser.add_argument('--no-eval_set__num_kpts_per_image',
    dest='eval_set__num_kpts_per_image',
    action='store_false',
    help='Do not add number of keypoints per image for each set.',
    default=None)

# Average number of found keypoints in all images in set.
parser.add_argument('--eval_set__num_kpts_per_image_avg',
    dest='eval_set__num_kpts_per_image_avg',
    action='store_true',
    help='Add average number of found keypoints in the images of the set. Default: True.',
    default=None)

parser.add_argument('--no-eval_set__num_kpts_per_image_avg',
    dest='eval_set__num_kpts_per_image_avg',
    action='store_false',
    help='Do not add average number of found keypoints in the imgaes of the set.',
    default=None)

# Standard deviation of found keypoints in all images in  set.
parser.add_argument('--eval_set__num_kpts_per_image_std',
    dest='eval_set__num_kpts_per_image_std',
    action='store_true',
    help='Add standard deviation of average number of keypoints in the images of the set. Default: True.',
    default=None)

parser.add_argument('--no-eval_set__num_kpts_per_image_std',
    dest='eval_set__num_kpts_per_image_std',
    action='store_false',
    help='Do not add standard deviation average number of keypoints in the images of the set.',
    default=None)

# Names of image files in set.
parser.add_argument('--eval_set__image_names',
    dest='eval_set__image_names',
    action='store_true',
    help='Add the image file names for each set.',
    default=None)

parser.add_argument('--no-eval_set__image_names',
    dest='eval_set__image_names',
    action='store_false',
    help='Do not add the image file names for each set.',
    default=None)

# Repeatable keypoints
parser.add_argument('--eval_set__num_repeatable_kpts',
    dest='eval_set__num_repeatable_kpts',
    action='store_true',
    help='Add the number of repeatable keypoints over all images in set. Default: True.',
    default=None)

parser.add_argument('--no-eval_set__num_repeatable_kpts',
    dest='eval_set__num_repeatable_kpts',
    action='store_false',
    help='Do not add the number of repeatable keypoints over all images in set.',
    default=None)

# Indices of repeatable keypoints
parser.add_argument('--eval_set__idx_repeatable_kpts',
    dest='eval_set__idx_repeatable_kpts',
    action='store_true',
    help='Add the indices of the repeatable keypoints over all images in set. Default: True.',
    default=None)

parser.add_argument('--no-eval_set__idx_repeatable_kpts',
    dest='eval_set__idx_repeatable_kpts',
    action='store_false',
    help='Do not add the indices of the repeatable keypoints over all images in set.',
    default=None)

# Cumulative repeatability
parser.add_argument('--eval_set__cum_repeatable_kpts',
    dest='eval_set__cum_repeatable_kpts',
    action='store_true',
    help='Add number of repeatable keypoints up to n-th image for each image in set. Default: True.',
    default=None)

parser.add_argument('--no-eval_set__cum_repeatable_kpts',
    dest='eval_set__cum_repeatable_kpts',
    action='store_false',
    help='Do not add number of repeatable keypoints up to n-th image for each image in set.',
    default=None)

# Switch all evaluations ON/OFF:
parser.add_argument('--eval_set__enable_all',
    dest='eval_set__enable_all',
    action='store_true',
    help='Enable all evaluation tests. Default: True.',
    default=True)

parser.add_argument('--eval_set__disable_all',
    dest='eval_set__enable_all',
    action='store_false',
    help='Disable all evaluation tests')

table_detector_names = {
    'sift': 'SIFT',
    'lift': 'LIFT',
    'superpoint': 'SuperPoint',
    'tilde': 'TILDE'
}

table_eval_set = {
    'eval_set__num_kpts_per_image': True,
    'eval_set__num_kpts_per_image_avg': True,
    'eval_set__num_kpts_per_image_std': True,
    'eval_set__image_names': True,
    'eval_set__num_repeatable_kpts': True,
    'eval_set__idx_repeatable_kpts': True,
    'eval_set__cum_repeatable_kpts': True
}

def get_config(argv):
    config, _ = parser.parse_known_args()
    config = vars(config)

    # If enable_all is not true, only take the evaluation tests, that have
    # specifically been activated.
    if not config['eval_set__enable_all']:
        for key in list(table_eval_set.keys()):
            table_eval_set[key] = config[key] if config[key] else False

    if config['eval_set__enable_all']:
        for key in list(table_eval_set.keys()):
            table_eval_set[key] = False if config[key] == False else True

    # Finally update the config object.
    for key in list(table_eval_set.keys()):
        config[key] = table_eval_set[key]

    config['detector_names'] = [table_detector_names[x] for x in config['model_names']]
    config['kpts_file_format'] = 'kpts_{}__{}_{}.csv' \
        if config['max_size'] is None else 'kpts_{}__{}_{}_{}.csv'

    return config
