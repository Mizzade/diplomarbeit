"""
Run trained detector with image to get feature map

Usage: python patch_network_test_point.py --train_name mexico_tilde_p24_Mexico_train_point_translation_iter_20 --stats_name mexico_tilde_p24_Mexico_train_point --dataset_name VggAffineDataset --save_feature covariant_point_tilde

General options:
    --training      Name of the training dataset
    --stats_name    Name of the mean and the variance of the patches
    --dataset_name  dataset name (option VggAffineDataset, EFDataset, WebcamDataset)
    --save_feature  name to save the feature
    --alpha         Trade-off parameter between invertible loss and covariant loss
    --descriptor_dim Number of the parameter for transformation (translation 2)
    --patch_size    Default 32

Output:

    feature map

Examples:

    >>> python patch_network_test_point.py --train_name mexico_tilde_p24_Mexico_train_point_translation_iter_20 --stats_name mexico_tilde_p24_Mexico_train_point --dataset_name VggAffineDataset --save_feature covariant_point_tilde

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

parser.add_argument("--train_name", nargs='?', type=str, default = 'mexico_tilde_p24_Mexico_train_point_translation_iter_20',
                    help="Training dataset name")

parser.add_argument("--stats_name", nargs='?', type=str, default = 'mexico_tilde_p24_Mexico_train_point',
                    help="Training dataset name")

parser.add_argument("--dataset_name",
    nargs='?',
    type=str,
    default='webcam',
    help="Training dataset name. Name of the image collection.")

parser.add_argument("--save_feature", nargs='?', type=str, default = 'covariant_point_tilde',
                    help="Training dataset name")

parser.add_argument("--alpha", nargs='?', type=float, default = 1.0,
                    help="alpha")

parser.add_argument("--descriptor_dim", nargs='?', type=int, default = 2,
                    help="Number of embedding dimemsion")

parser.add_argument("--patch_size", nargs='?', type=int, default = 32,
                    help="Size of the patch")

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

args = parser.parse_args()
train_name = args.train_name
stats_name = args.stats_name
dataset_name = args.dataset_name
save_feature_name = args.save_feature

output_dir = args.output_dir
image_dir = args.image_dir
data_dir = args.data_dir

tensorflow_dir = os.path.join(data_dir, 'tensorflow')
stats_dir = os.path.join(data_dir, 'stats')

# Parameters
patch_size = args.patch_size
batch_size = 128
descriptor_dim = args.descriptor_dim

print('Loading training stats:')


with open(os.path.join(stats_dir, 'stats_{}.pkl'.format(stats_name)), 'rb') as src:
    mean, std = pickle.load(src, encoding='utf-8')
print(mean)
print(std)

CNNConfig = {
    "patch_size": patch_size,
    "descriptor_dim" : descriptor_dim,
    "batch_size" : batch_size,
    "alpha" : args.alpha,
    "train_flag" : False
}

cnn_model = patch_cnn.PatchCNN(CNNConfig)

#dataset information
# subsets = []
# if dataset_name=='VggAffineDataset' :
#     subsets = ['bikes', 'trees', 'graf', 'wall', 'boat', 'bark', 'leuven', 'ubc']

# if dataset_name=='EFDataset' :
#     subsets = ['notredame','obama','paintedladies','rushmore','yosemite']

if dataset_name=='webcam' :
    subsets = ['chamonix', 'courbevoie', 'frankfurt', 'panorama', 'stlouis'] #'Mexico', #not using mexico for test,

working_dir = data_dir
load_dir = os.path.join(image_dir, dataset_name)
save_dir = os.path.join(output_dir, save_feature_name, dataset_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    try:
        saver.restore(sess, os.path.join(tensorflow_dir, '{}_model.ckpt'.format(train_name)))
        # saver.restore(sess, "../tensorflow_model/"+train_name+"_model.ckpt")
        print("Model restored.")
    except:
        print('No model found')
        exit()

    for i,subset in enumerate(subsets) :
        index = 1

        # Create output dir for each data set
        if not os.path.exists(os.path.join(save_dir, subset)):
            os.makedirs(os.path.join(save_dir, subset), exist_ok=True)

        for file in os.listdir(os.path.join(load_dir, subset)):
            output_list = []
            if file.endswith(".ppm") or file.endswith(".pgm") or file.endswith(".png") or file.endswith(".jpg") :
                image_name = os.path.join(load_dir, subset, file)
                print(image_name)
                save_file = file[0:-4] + '.mat'
                save_name = os.path.join(save_dir, subset, save_file)

                #read image
                img, ratio = read_image_from_name(image_name)
                if img.shape[2] == 1 :
                    img = np.repeat(img, 3, axis = 2)

                #build image pyramid
                pyramid = pyramid_gaussian(img, max_layer = 4, downscale=np.sqrt(2))

                #predict transformation
                for (j, resized) in enumerate(pyramid) :
                    fetch = {
                        "o1": cnn_model.o1
                    }

                    resized = np.asarray(resized)
                    resized = (resized-mean)/std
                    resized = resized.reshape((1,resized.shape[0],resized.shape[1],resized.shape[2]))

                    result = sess.run(fetch, feed_dict={cnn_model.patch: resized})
                    result_mat = result["o1"].reshape((result["o1"].shape[1],result["o1"].shape[2],result["o1"].shape[3]))
                    output_list.append(result_mat)

                sio.savemat(save_name,{'output_list':output_list})
