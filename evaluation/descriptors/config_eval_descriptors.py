import argparse
import os
import sys

"""Beinhaltet alle Konfigurationen für die Evaluation von Deskriptoren."""

parser = argparse.ArgumentParser()
parser.add_argument('--dry',
    dest='dry',
    action='store_true',
    help='If set, only print config, but do not run evaluation.',
    default=False)

parser.add_argument('--model_combinations', nargs='+',
    help='List of detectors and descriptors to be evaluated together. ' +
    'E.g. "--model_combinations tilde sift" evaluates the sift descriptor ' +
    'with the tilde detector.',
    default=['sift', 'sift'])

parser.add_argument('--allowed_extensions', nargs='+',
    help='Set the allowed file extensions for the imag directory.',
    default=['.png'])

parser.add_argument('--output_file_prefix', type=str,
    help='Prefix for output file name.',
    default='eval_')

parser.add_argument('--max_size',type=str,
    help='Look for files with this max size postfix.',
    default='1300')

parser.add_argument('--collection_names', nargs='+',
    help='Names of all collections to use.',
    default=['webcam'])

parser.add_argument('--set_names', nargs='+',
    help='Names of all sets to be used.',
    default=['chamonix', 'courbevoie', 'frankfurt', 'mexico', 'panorama', 'stlouis'])

# Directory entries
parser.add_argument('--images_dir', type=str,
    help='Path to the image directory containing the collections.',
    default='../../data')

parser.add_argument('--data_dir', type=str,
    help='Path to the data dir containing the outputs of the detectors and descriptors.',
    default='../../outputs')

parser.add_argument('--output_dir', type=str,
    help='Path to where to store this evaluation outputs.',
    default='outputs')

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

# Evaluations

parser.add_argument('--eval_image_pair__num_max_possible_matches',
    dest='eval_image_pair__num_max_possible_matches',
    action='store_true',
    help='Evaluate the number of maximal possible matches between image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__num_max_possible_matches',
    dest='eval_image_pair__num_max_possible_matches',
    action='store_false',
    help='Do notEvaluate the number of maximal possible matches between image pair.',
    default=None)

parser.add_argument('--eval_image_pair__num_found_matches',
    dest='eval_image_pair__num_found_matches',
    action='store_true',
    help='Evaluate the number of found matches between image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__num_found_matches',
    dest='eval_image_pair__num_found_matches',
    action='store_false',
    help='Do not evaluate the number of found matches between image pair.',
    default=None)

parser.add_argument('--eval_image_pair__perc_found_matches',
    dest='eval_image_pair__perc_found_matches',
    action='store_true',
    help='Evaluate the percentage of found matches of all possible matches.',
    default=None)

parser.add_argument('--no-eval_image_pair__perc_found_matches',
    dest='eval_image_pair__perc_found_matches',
    action='store_false',
    help='Do not evaluate the percentage of found matches of all possible matches.',
    default=None)

parser.add_argument('--eval_image_pair__num_matches_ratio_test',
    dest='eval_image_pair__num_matches_ratio_test',
    action='store_true',
    help='Evaluate the number of matches that pass the ratio test.',
    default=None)

parser.add_argument('--no-eval_image_pair__num_matches_ratio_test',
    dest='eval_image_pair__num_matches_ratio_test',
    action='store_false',
    help='Do not evaluate the number of matches that pass the ratio test.',
    default=None)

parser.add_argument('--eval_image_pair__perc_matches_ratio_test',
    dest='eval_image_pair__perc_matches_ratio_test',
    action='store_true',
    help='Evaluate the percentage of matches that pass the ratio test.',
    default=None)

parser.add_argument('--no-eval_image_pair__perc_matches_ratio_test',
    dest='eval_image_pair__perc_matches_ratio_test',
    action='store_false',
    help='Do not evaluate the percentage of matches that pass the ratio test.',
    default=None)

parser.add_argument('--eval_image_pair__num_inlier_fmatrix',
    dest='eval_image_pair__num_inlier_fmatrix',
    action='store_true',
    help='Evaluate the number of matches that are inlier for estimation of fundamental matrix.',
    default=None)

parser.add_argument('--no-eval_image_pair__num_inlier_fmatrix',
    dest='eval_image_pair__num_inlier_fmatrix',
    action='store_false',
    help='Do not evaluate the number of matches that are inlier for estimation of fundamental matrix.',
    default=None)

parser.add_argument('--eval_image_pair__perc_inlier_fmatrix',
    dest='eval_image_pair__perc_inlier_fmatrix',
    action='store_true',
    help='Evaluate the percentage of matches that are inlier for estimation of fundamental matrix.',
    default=None)

parser.add_argument('--no-eval_image_pair__perc_inlier_fmatrix',
    dest='eval_image_pair__perc_inlier_fmatrix',
    action='store_false',
    help='Do not evaluate the percentage of matches that are inlier for estimation of fundamental matrix.',
    default=None)

parser.add_argument('--eval_image_pair__perc_inlier_ratio_test_fmatrix',
    dest='eval_image_pair__perc_inlier_ratio_test_fmatrix',
    action='store_true',
    help='Evaluate the percentage of matches that passes the ratio test and that are inlier for estimation of fundamental matrix.',
    default=None)

parser.add_argument('--no-eval_image_pair__perc_inlier_ratio_test_fmatrix',
    dest='eval_image_pair__perc_inlier_ratio_test_fmatrix',
    action='store_false',
    help='Do not evaluate the percentage of matches that passes the ratio test and that are inlier for estimation of fundamental matrix.',
    default=None)

parser.add_argument('--eval_image_pair__sum_dist_epilines_first_image',
    dest='eval_image_pair__sum_dist_epilines_first_image',
    action='store_true',
    help='Evaluate the distance in pixel of keypoints in the first image to the epipolar lines of keypoints in the second image a an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__sum_dist_epilines_first_image',
    dest='eval_image_pair__sum_dist_epilines_first_image',
    action='store_false',
    help='Do not evaluate the distance in pixel of keypoints in the first image to the epipolar lines of keypoints in the second image a an image pair.',
    default=None)

parser.add_argument('--eval_image_pair__mean_dist_epilines_first_image',
    dest='eval_image_pair__mean_dist_epilines_first_image',
    action='store_true',
    help='Evaluate the mean distance in pixel of keypoints in the first image to the epipolar lines of keypoints in the second image a an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__mean_dist_epilines_first_image',
    dest='eval_image_pair__mean_dist_epilines_first_image',
    action='store_false',
    help='Do not evaluate the mean distance in pixel of keypoints in the first image to the epipolar lines of keypoints in the second image a an image pair.',
    default=None)

parser.add_argument('--eval_image_pair__std_dist_epilines_first_image',
    dest='eval_image_pair__std_dist_epilines_first_image',
    action='store_true',
    help='Evaluate the standard deviation of the distance in pixel of keypoints in the first image to the epipolar lines of keypoints in the second image a an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__std_dist_epilines_first_image',
    dest='eval_image_pair__std_dist_epilines_first_image',
    action='store_false',
    help='Do not evaluate the standard deviation of the distance in pixel of keypoints in the first image to the epipolar lines of keypoints in the second image a an image pair.',
    default=None)

parser.add_argument('--eval_image_pair__sum_dist_epilines_second_image',
    dest='eval_image_pair__sum_dist_epilines_second_image',
    action='store_true',
    help='Evaluate the distance in pixel of keypoints in the second image to the epipolar lines of keypoints in the first image a an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__sum_dist_epilines_second_image',
    dest='eval_image_pair__sum_dist_epilines_second_image',
    action='store_false',
    help='Do not evaluate the distance in pixel of keypoints in the second image to the epipolar lines of keypoints in the first image a an image pair.',
    default=None)

parser.add_argument('--eval_image_pair__mean_dist_epilines_second_image',
    dest='eval_image_pair__mean_dist_epilines_second_image',
    action='store_true',
    help='Evaluate the mean distance in pixel of keypoints in the second image to the epipolar lines of keypoints in the first image a an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__mean_dist_epilines_second_image',
    dest='eval_image_pair__mean_dist_epilines_second_image',
    action='store_false',
    help='Do not evaluate the mean distance in pixel of keypoints in the second image to the epipolar lines of keypoints in the first image a an image pair.',
    default=None)

parser.add_argument('--eval_image_pair__std_dist_epilines_second_image',
    dest='eval_image_pair__std_dist_epilines_second_image',
    action='store_true',
    help='Evaluate the standard deviation of the  distance in pixel of keypoints in the second image to the epipolar lines of keypoints in the first image a an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__std_dist_epilines_second_image',
    dest='eval_image_pair__std_dist_epilines_second_image',
    action='store_false',
    help='Do not evaluate the standard deviation of the  distance in pixel of keypoints in the second image to the epipolar lines of keypoints in the first image a an image pair',
    default=None)

parser.add_argument('--eval_image_pair__sum_dist_epilines',
    dest='eval_image_pair__sum_dist_epilines',
    action='store_true',
    help='Evaluate the distance of all keypoints to all epipolar lines in both images of an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__sum_dist_epilines',
    dest='eval_image_pair__sum_dist_epilines',
    action='store_false',
    help='Do not evaluate the distance of all keypoints to all epipolar lines in both images of an image pair.',
    default=None)

parser.add_argument('--eval_image_pair__mean_dist_epilines',
    dest='eval_image_pair__mean_dist_epilines',
    action='store_true',
    help='Evaluate the mean distance of all keypoints to all epipolar lines in both images of an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__mean_dist_epilines',
    dest='eval_image_pair__mean_dist_epilines',
    action='store_false',
    help='Do not evaluate the mean  distance of all keypoints to all epipolar lines in both images of an image pair.',
    default=None)

parser.add_argument('--eval_image_pair__std_dist_epilines',
    dest='eval_image_pair__std_dist_epilines',
    action='store_true',
    help='Evaluate the standard deviation of the  distance of all keypoints to all epipolar lines in both images of an image pair.',
    default=None)

parser.add_argument('--no-eval_image_pair__std_dist_epilines',
    dest='eval_image_pair__std_dist_epilines',
    action='store_false',
    help='Do not evaluate the standard deviation of the  distance of all keypoints to all epipolar lines in both images of an image pair.',
    default=None)

# Evaluation tests
table_eval_set = {
    'eval_image_pair__num_max_possible_matches': True,
    'eval_image_pair__num_found_matches': True,
    'eval_image_pair__perc_found_matches': True,
    'eval_image_pair__num_matches_ratio_test': True,
    'eval_image_pair__perc_matches_ratio_test': True,
    'eval_image_pair__num_inlier_fmatrix': True,
    'eval_image_pair__perc_inlier_fmatrix': True,
    'eval_image_pair__perc_inlier_ratio_test_fmatrix': True,
    'eval_image_pair__sum_dist_epilines_first_image': True,
    'eval_image_pair__mean_dist_epilines_first_image': True,
    'eval_image_pair__std_dist_epilines_first_image': True,
    'eval_image_pair__sum_dist_epilines_second_image': True,
    'eval_image_pair__mean_dist_epilines_second_image': True,
    'eval_image_pair__std_dist_epilines_second_image': True,
    'eval_image_pair__sum_dist_epilines': True,
    'eval_image_pair__mean_dist_epilines': True,
    'eval_image_pair__std_dist_epilines': True
}

table_detector_names = {
    'sift': 'SIFT',
    'lift': 'LIFT',
    'superpoint': 'SuperPoint',
    'tilde': 'TILDE'
}

table_descriptor_names = {
    'sift': 'SIFT',
    'lift': 'LIFT',
    'superpoint': 'SuperPoint',
    'doap': 'DOAP',
    'tfeat': 'Tfeat'
}

def get_config(argv):
    config, _ = parser.parse_known_args()
    config = vars(config)

    if len(config['model_combinations']) % 2:
        sys.exit('Parameter <model_combinations> must be an even number! Exit.')

     # If enable_all is not true, only take the evaluation tests, that have
    # specifically been activated.
    if not config['eval_set__enable_all']:
        for key in list(table_eval_set.keys()):
            table_eval_set[key] = config[key] if config[key] else False

    # If enable_all is set, take all evaluation test except those, that have
    # been manually deactivated.
    if config['eval_set__enable_all']:
        for key in list(table_eval_set.keys()):
            table_eval_set[key] = False if config[key] == False else True

    # Finally update the config object.
    for key in list(table_eval_set.keys()):
        config[key] = table_eval_set[key]

    # Extract names of models, detectors and descriptors.
    config['model_names'] = [x for x in config['model_combinations'][1::2]]
    config['detector_names'] = [table_detector_names[x] for x in config['model_combinations'][::2]]
    config['descriptor_names'] = [table_descriptor_names[x] for x in config['model_combinations'][1::2]]

    config['desc_file_format'] = 'desc_{}__{}_{}_{}.csv' \
        if config['max_size'] is None else 'desc_{}__{}_{}_{}_{}.csv' 

    config['kpts_file_format'] = 'kpts_{}__{}_{}.csv' \
        if config['max_size'] is None else 'kpts_{}__{}_{}_{}.csv'


    return config