import argparse
import os

# MAIN PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--allowed_extensions', nargs='+',
    help='Set the allowed file extensions for the image directory.',
    default=['.jpg', '.png', '.ppm', '.jpeg', '.tiff'])

parser.add_argument('--root_dir', type=str,
    help='Set the path to the root directory of this repository.',
    required=True)

parser.add_argument('--output_dir', type=str,
    help='Path to directory containing the results of the models. Relative to root_dir',
    default='outputs')

parser.add_argument('--data_dir', type=str,
    help='Path to the directory containing all image sets. Relative to root_dir',
    default='data')

parser.add_argument('--size', type=int, default=None,
    help='Set to the maximal dimension size every image should have.')

parser.add_argument('--networks', nargs='+',
    help='Set the networks to run when calling run_moles.py.',
    default=['sift', 'superpoint', 'tfeat', 'doap', 'lift', 'tilde'])

parser.add_argument('--max_num_images', type=int, default=None,
    help='Maximum number of images to be used by networks. If not set, take all images found.')

parser.add_argument('--skip_first_n', type=int, default=None,
    help='Skip the first n images when using a model.')

parser.add_argument('--collection_names', nargs='+',
    help='Name of all collections to be processed within the data_dir. Skip collections that are not found. A value of "None" means all collections.',
    default=['webcam', 'eisert'])

parser.add_argument('--set_names', nargs='+',
    help='Name of all sets to be processed within a collection. Skip sets that are not found. A value of "None" means all sets.',
    default=None)

parser.add_argument('--dry',
    dest='dry',
    action='store_true',
    help='If set, only print config, but do not run evaluation.',
    default=False)

subparsers = parser.add_subparsers(help='Commands for the different networks.')
# SIFT
parser_sift = subparsers.add_parser('sift', help='Configurations for SIFT.')
parser_sift.add_argument('--name', type=str, default='SIFT',
    help='Name of the SIFT model.')

parser_sift.add_argument('--dir', type=str, default='pipe_sift',
    help='Path to dir containing the sift files relative from root_dir.')

parser_sift.add_argument('--main', type=str, default='use_sift.py',
    help='The main file to start the SIFT model.')

parser_sift.add_argument('--tmp_dir', type=str, default=None,
    help='Path to the temporary dir to save intermediate results. Relative to --dir.')

# SuperPoint
parser_superpoint = subparsers.add_parser('superpoint', help='Configurations for SuperPoint.')
parser_superpoint.add_argument('--name', type=str, default='SuperPoint',
    help='Name of the SuperPoint model.')

parser_superpoint.add_argument('--dir', type=str, default='pipe_superpoint',
    help='Path to dir containing the SuperPoint files relative from root_dir.')

parser_superpoint.add_argument('--main', type=str, default='use_superpoint.py',
    help='The main file to start the SuperPoint model.')

parser_superpoint.add_argument('--tmp_dir', type=str, default=None,
    help='Path to the temporary dir to save intermediate results. Relative to --dir.')

# Tfeat
parser_tfeat = subparsers.add_parser('tfeat', help='Configurations for Tfeat.')
parser_tfeat.add_argument('--name', type=str, default='Tfeat',
    help='Name of the Tfeat model.')

parser_tfeat.add_argument('--dir', type=str, default='desc_tfeat',
    help='Path to dir containing the Tfeat files relative from root_dir.')

parser_tfeat.add_argument('--main', type=str, default='use_tfeat.py',
    help='The main file to start the Tfeat model.')

parser_tfeat.add_argument('--tmp_dir', type=str, default=None,
    help='Path to the temporary dir to save intermediate results. Relative to --dir.')

# DOAP
parser_doap = subparsers.add_parser('doap', help='Configurations for DOAP.')
parser_doap.add_argument('--name', type=str, default='DOAP',
    help='Name of the DOAP model.')

parser_doap.add_argument('--dir', type=str, default='desc_doap',
    help='Path to dir containing the DOAP files relative from root_dir.')

parser_doap.add_argument('--main', type=str, default='use_doap.py',
    help='The main file to start the DOAP model.')

parser_doap.add_argument('--tmp_dir', type=str, default='tmp',
    help='Path to the temporary dir to save intermediate results. Relative to --dir.')

# LIFT
parser_lift = subparsers.add_parser('lift', help='Configurations for LIFT.')
parser_lift.add_argument('--name', type=str, default='LIFT',
    help='Name of the LIFT model.')

parser_lift.add_argument('--dir', type=str, default='pipe_lift',
    help='Path to dir containing the LIFT files relative from root_dir.')

parser_lift.add_argument('--main', type=str, default='use_lift.py',
    help='The main file to start the LIFT model.')

parser_lift.add_argument('--tmp_dir', type=str, default='tmp',
    help='Path to the temporary dir to save intermediate results. Relative to --dir.')

# TILDE
parser_tilde = subparsers.add_parser('tilde', help='Configurations for TILDE.')
parser_tilde.add_argument('--name', type=str, default='TILDE',
    help='Name of the TILDE model.')

parser_tilde.add_argument('--dir', type=str, default='det_tilde',
    help='Path to dir containing the TILDE files relative from root_dir.')

parser_tilde.add_argument('--main', type=str, default='use_tilde.sh',
    help='The main file to start the TILDE model.')

parser_tilde.add_argument('--tmp_dir', type=str, default='tmp',
    help='Path to the temporary dir to save intermediate results. Relative to --dir.')

subparser_table = {
    'sift': parser_sift,
    'superpoint': parser_superpoint,
    'tfeat': parser_tfeat,
    'doap': parser_doap,
    'lift': parser_lift,
    'tilde': parser_tilde

}

def get_config(argv):
    config, _ = parser.parse_known_args()
    networks = [vars(subparser_table[x].parse_known_args()[0]) for x in config.networks]
    networks = [{**x,
        'dir': os.path.join(config.root_dir, x['dir']),
        'tmp_dir': os.path.join(config.root_dir, x['dir'], x['tmp_dir']) if x['tmp_dir'] is not None else None} for x in networks]

    config.output_dir = os.path.join(config.root_dir, config.output_dir)
    config.data_dir = os.path.join(config.root_dir, config.data_dir)

    return config, networks
