import argparse
import os
import run_support_functions as rsf

parser = argparse.ArgumentParser()

parser.add_argument('--configuration_name',
    type=str,
    help='Name of this configiration. This will be the title if DRY is set.',
    default='Configuration: run_descriptors.py')

parser.add_argument('--verbose',
    dest='verbose',
    action='store_true',
    help='Allow for more informational output while models working.',
    default=False)

parser.add_argument('--task',
    type=str,
    choices=['descriptors', 'patches'],
    help='Select a task for descriptors: descriptors or patches. ' +
    'Default: descriptors',
    default='descriptors')

parser.add_argument('--root_dir',
    type=str,
    help='Set the path to the root directory of this repository.',
    required=True)

parser.add_argument('--allowed_extensions',
    nargs='+',
    help='Set the allowed file extensions for the image directory. Only files ' +
    'with the fitting extension will be used by the detector. Default: ' +
    '(.jpg, .png, .ppm, .jpeg, .tiff)',
    default=['.jpg', '.png', '.ppm', '.jpeg', '.tiff'])

parser.add_argument('--output_dir',
    type=str,
    help='Path to directory containing the results of the detectors. ' +
    'Relative to ROOT_DIR. Default: outputs',
    default='outputs')

parser.add_argument('--data_dir',
    type=str,
    help='Path to the directory containing all image sets. Relative to ' +
    'ROOT_DIR. Default: data',
    default='data')

parser.add_argument('--max_size',
    type=int,
    help='Set to the maximal dimension size every image can have. Forces a ' +
    'resizing of images, that are larger than this size. A value of None ' +
    'means to take the image with its original size. Default: None',
    default=None)

parser.add_argument('--descriptors',
    nargs='+',
    help='Choose which descriptors should be run. Default: (sift, tfeat, doap, lift)',
    default=['sift', 'tfeat', 'doap', 'lift'])

parser.add_argument('--detectors',
    nargs='+',
    help='Choose which detectors should be run. Default: (sift, tilde, lift, tcovdet, superpoint)',
    default=['sift', 'tilde', 'lift', 'tcovdet', 'superpoint'])

parser.add_argument('--max_num_images',
    type=int,
    help='Maximum number of images to be used by A model. A value of None ' +
    'means to take all images found. Default: None',
    default=None)

parser.add_argument('--skip_first_n',
    type=int,
    help='Skip the first n images when using a model. A value of None means ' +
    'not to skip any images. Default: None',
    default=None)

parser.add_argument('--collection_names',
    nargs='+',
    help='Name of all collections to be processed within the data_dir. ' +
    'Skip collections that are not found. A value of None means all ' +
    'collections. Default: None',
    default=None)

parser.add_argument('--set_names',
    nargs='+',
    help='Name of all sets to be processed within the data_dir. ' +
    'Skip sets that are not found. A value of None means all sets. ' +
    'Default: None',
    default=None)

parser.add_argument('--dry',
    dest='dry',
    action='store_true',
    help='If set, only print config, but do not run models. Default: False',
    default=False)

# Arguments for descriptors
# SIFT
parser.add_argument('--root_dir_sift',
    type=str,
    help='Path to the root folder of the SIFT module relative from ROOT_DIR. ' +
    'Default: pipe_sift',
    default='pipe_sift')

parser.add_argument('--tmp_dir_sift',
    type=str,
    help='Path to temporary directory to save intermediate results. Relative ' +
    'to ROOT_DIR_SIFT. Default: tmp',
    default='tmp')

parser.add_argument('--main_sift',
    type=str,
    help='Name of the main python file to start the model. Default: use_sift.py',
    default='use_sift.py')

# TFeat
parser.add_argument('--root_dir_tfeat',
    type=str,
    help='Path to the root folder of the TFeat module relative from ROOT_DIR. ' +
    'Default: desc_tfeat',
    default='desc_tfeat')

parser.add_argument('--tmp_dir_tfeat',
    type=str,
    help='Path to temporary directory to save intermediate results. Relative ' +
    'to ROOT_DIR_TFEAT. Default: tmp',
    default='tmp')

parser.add_argument('--main_tfeat',
    type=str,
    help='Name of the main python file to start the model. Default: use_tfeat.py',
    default='use_tfeat.py')

# DOAP
parser.add_argument('--root_dir_doap',
    type=str,
    help='Path to the root folder of the TFeat module relative from ROOT_DIR. ' +
    'Default: desc_doap',
    default='desc_doap')

parser.add_argument('--tmp_dir_doap',
    type=str,
    help='Path to temporary directory to save intermediate results. Relative ' +
    'to ROOT_DIR_DOAP. Default: tmp',
    default='tmp')

parser.add_argument('--main_doap',
    type=str,
    help='Name of the main python file to start the model. Default: use_doap.py',
    default='use_doap.py')

# LIFT
parser.add_argument('--root_dir_lift',
    type=str,
    help='Path to the root folder of the LIFT module relative from ROOT_DIR. ' +
    'Default: pipe_lift',
    default='pipe_lift')

parser.add_argument('--tmp_dir_lift',
    type=str,
    help='Path to temporary directory to save intermediate results. Relative ' +
    'to ROOT_DIR_LIFT. Default: tmp',
    default='tmp')

parser.add_argument('--main_lift',
    type=str,
    help='Name of the main python file to start the model. Default: use_lift.py',
    default='use_lift.py')

# SuperPoint
parser.add_argument('--root_dir_superpoint',
    type=str,
    help='Path to the root folder of the SuperPoint module relative from ROOT_DIR. ' +
    'Default: pipe_superpoint',
    default='pipe_superpoint')

parser.add_argument('--tmp_dir_superpoint',
    type=str,
    help='Path to temporary directory to save intermediate results. Relative ' +
    'to ROOT_DIR_SUPERPOINT. Default: tmp',
    default='tmp')

parser.add_argument('--main_superpoint',
    type=str,
    help='Name of the main python file to start the model. Default: use_superpoint.py',
    default='use_superpoint.py')



def get_config(argv):
    config, _ = parser.parse_known_args()
    config = vars(config)

    config['collection_names'] = rsf.get_collection_names(config)
    config['set_names'] = rsf.get_set_names(config)

    # Set absolute paths
    config['data_dir'] = os.path.join(config['root_dir'], config['data_dir'])
    config['output_dir'] = os.path.join(config['root_dir'], config['output_dir'])

    config['root_dir_sift'] = os.path.join(config['root_dir'], config['root_dir_sift'])
    config['tmp_dir_sift'] = os.path.join(config['root_dir_sift'], config['tmp_dir_sift'])

    config['root_dir_tfeat'] = os.path.join(config['root_dir'], config['root_dir_tfeat'])
    config['tmp_dir_tfeat'] = os.path.join(config['root_dir_tfeat'], config['tmp_dir_tfeat'])

    config['root_dir_doap'] = os.path.join(config['root_dir'], config['root_dir_doap'])
    config['tmp_dir_doap'] = os.path.join(config['root_dir_doap'], config['tmp_dir_doap'])

    config['root_dir_lift'] = os.path.join(config['root_dir'], config['root_dir_lift'])
    config['tmp_dir_lift'] = os.path.join(config['root_dir_lift'], config['tmp_dir_lift'])

    config['root_dir_superpoint'] = os.path.join(config['root_dir'], config['root_dir_superpoint'])
    config['tmp_dir_superpoint'] = os.path.join(config['root_dir_superpoint'], config['tmp_dir_superpoint'])

    return config





