"""
Run trained detector with image to get feature map
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import gc
import patch_reader
import patch_cnn
import numpy as np
from scipy.spatial import distance
import scipy.io as sio
import glob
import pickle
import cv2
import os
from skimage.transform import pyramid_gaussian
import exifread
import argparse
import sys
import io_utils

from datetime import datetime

def read_image_from_name(file_name):
    """Return the factorial of n, an exact integer >= 0. Image rescaled to no larger than 1024*768

    Args:
        root_dir (str): Image directory
        filename (str): Name of the file

    Returns:
        np array: Image
        float: Rescale ratio

    """
    img = cv2.imread(file_name)
    if img.shape[2] == 4 :
        img = img[:,:,:3]

    if img.shape[2] == 1 :
            img = np.repeat(img, 3, axis = 2)

    ftest = open(file_name, 'rb')
    tags = exifread.process_file(ftest)

    try:
        if str(tags['Thumbnail Orientation']) == 'Rotated 90 CW':
            img = cv2.transpose(img)
            img = cv2.flip(img, 1)
        elif str(tags['Thumbnail Orientation']) == 'Rotated 90 CCW':
            img = cv2.transpose(img)
            img = cv2.flip(img, 0)
        elif str(tags['Thumbnail Orientation']) == 'Rotated 180':
            img = cv2.flip(img, -1)
    except:
        tags = tags

    ratio = 1.0
    if img.shape[0]*img.shape[1]>1024*768:
        ratio = (1024*768/float(img.shape[0]*img.shape[1]))**(0.5)
        img = cv2.resize(img,(int(img.shape[1]*ratio), int(img.shape[0]*ratio)),interpolation = cv2.INTER_CUBIC);

    return img, ratio

parser = argparse.ArgumentParser()

parser.add_argument('--train_name',
    nargs='?',
    type=str,
    default='mexico_tilde_p24_Mexico_train_point_translation_iter_20',
    help='Name of the tensorflow model to use within the TENSORFLOW_DIR. ' +
    'Default: mexico_tilde_p24_Mexico_train_point_translation_iter_20')

parser.add_argument('--stats_name',
    nargs='?',
    type=str,
    default='mexico_tilde_p24_Mexico_train_point',
    help='Name of the stats file containing mean and std for the training data ' +
    'used to create the tensorflow model. Default: mexico_tilde_p24_Mexico_train_point')

parser.add_argument('--save_feature',
    nargs='?',
    type=str,
    default ='covariant_point_tilde',
    help='Name of the subfolder within OUTPUT_DIR wherein to save the ' +
    'covariant features. Default: covariant_point_tilde.')

parser.add_argument('--alpha',
    nargs='?',
    type=float,
    default = 1.0,
    help='Learning rate alpha. Default: 1.0')

parser.add_argument('--descriptor_dim',
    nargs='?',
    type=int,
    default=2,
    help='Number of embedding dimemsions. Default: 2')

parser.add_argument('--patch_size',
    nargs='?',
    type=int,
    default=32,
    help='Size of the patches. Default: 32')

parser.add_argument('--batch_size',
    nargs='?',
    type=int,
    default=128,
    help='Batch size when computing features. Default: 128')

parser.add_argument('--output_dir',
    type=str,
    help='Output directory. Relative to this file. Default: ./output',
    default='output')

parser.add_argument('--image_dir',
    type=str,
    help='Directory containing the image collections. Default: ./images',
    default='images')

parser.add_argument('--data_dir',
    type=str,
    help='Directory containing the stats and and the tensorflow dirs. ' +
    'Default: data',
    default='data')

parser.add_argument('--tensorflow_dir',
    type=str,
    help='Folder containing the .ckpt files for the tensorflow model. ' +
    'Relative to DATA_DIR. Default: tensorflow',
    default='tensorflow')

parser.add_argument('--stats_dir',
    type=str,
    help='Folder containing the stas (mean, std) for the training data used ' +
    'to create the model in tensorflow. Relative to DATA_DIR. Default: stats',
    default='stats')

parser.add_argument('--file_list',
    type=str,
    help='List of absolute file paths to images for which to compute keypoints ' +
    ' as a long string.')

parser.add_argument('--dry',
    dest='dry',
    action='store_true',
    help='If checked, only print parsed args and return. Default: False',
    default=False)

def main(args):
    # Get necessary directories
    output_dir = args.output_dir
    image_dir = args.image_dir
    data_dir = args.data_dir
    tensorflow_dir = os.path.join(data_dir, args.tensorflow_dir)
    stats_dir = os.path.join(data_dir, args.stats_dir)

    train_name = args.train_name
    stats_name = args.stats_name
    save_feature_name = args.save_feature

    # Create list of file names.
    file_list = args.file_list.split(' ')

    # Parameters
    alpha = args.alpha
    patch_size = args.patch_size
    batch_size = args.batch_size
    descriptor_dim = args.descriptor_dim

    stats_path = os.path.join(stats_dir, 'stats_{}.pkl'.format(stats_name))



    print('Loading training stats:')
    with open(stats_path, 'rb') as src:
        mean, std = pickle.load(src, encoding='utf-8')
    print('Training data loaded:\nMean: {}\nStd: {}'.format(mean, std))

    CNNConfig = {
        'patch_size': patch_size,
        'descriptor_dim' : descriptor_dim,
        'batch_size' : batch_size,
        'alpha' : alpha,
        'train_flag' : False
    }

    cnn_model = patch_cnn.PatchCNN(CNNConfig)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        try:
            saver.restore(sess, os.path.join(tensorflow_dir, '{}_model.ckpt'.format(train_name)))
            # saver.restore(sess, '../tensorflow_model/'+train_name+'_model.ckpt')
            print('Model restored.')
        except:
            print('No model found .Exit.')
            exit()


        for file_path in file_list:
            collection_name, set_name, file_base, extension = io_utils.get_path_components(file_path)
            file_name = '{}.{}'.format(file_base, extension)

            save_dir = os.path.join(output_dir, save_feature_name, collection_name, set_name)
            io_utils.create_dir(save_dir) # Create folder if it does not exists

            output_list = []
            if extension in ['.ppm', '.pgm', '.png', '.jpg']:
                save_file = file_base + '.mat'
                save_name = os.path.join(save_dir, save_file)

                #read image
                img, ratio = read_image_from_name(file_path)
                if img.shape[2] == 1 :
                    img = np.repeat(img, 3, axis = 2)

                #build image pyramid
                pyramid = pyramid_gaussian(img, max_layer = 4, downscale=np.sqrt(2))

                #predict transformation
                for (j, resized) in enumerate(pyramid) :
                    fetch = {
                        'o1': cnn_model.o1
                    }

                    resized = np.asarray(resized)
                    resized = (resized-mean)/std
                    resized = resized.reshape((1,resized.shape[0],resized.shape[1],resized.shape[2]))

                    result = sess.run(fetch, feed_dict={cnn_model.patch: resized})
                    result_mat = result['o1'].reshape((result['o1'].shape[1],result['o1'].shape[2],result['o1'].shape[3]))
                    output_list.append(result_mat)

                sio.savemat(save_name,{'output_list':output_list})

if __name__ == '__main__':
    args = parser.parse_args()

    if args.dry:
        for k, v in vars(args).items():
            print('{}: {}'.format(k, v))
    else:
        main(args)
