import scipy.io as sio
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import h5py
from typing import List, Dict, Any
from tqdm import tqdm
import argparse

def warpPerspectiveNoCrop(image:np.array, M:np.array, flags=cv2.INTER_AREA):
    # Height and  width of current image
    h, w = image.shape[:2]

    # Corners points of the current image
    corners = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]]).reshape(-1, 1, 2)

    # Corners after warping
    corners_warped = cv2.perspectiveTransform(corners, M)

    # Find the bounding rectangle
    bx, by, bwidth, bheight = cv2.boundingRect(corners_warped)

    #res = cv2.warpPerspective(image, Ht.dot(M), (xmax-xmin, ymax-ymin))
    bx, by, bwidth, bheight

    # Compute the translation homography tha will move (bx, by) to (0, 0)
    th = np.array([
        [1, 0, -bx],
        [0, 1, -by],
        [0, 0, 1]
    ])

    # Combines the homographies
    A = th.dot(M)

    # Apply the transformation to the image
    res = cv2.warpPerspective(image, A, (bwidth, bheight), flags=flags)
    return res


def create_training_pairs(
    num_patches:int,
    patch_type:str,             # test | training
    offset:int,                 # 5000
    training_offset:int,
    stride:int,                 # 1
    feature_dim:int,            # 2
    num_channels:int,           # 3
    patches:np.array,
    patch_size:int,
    fn_im:str,
    fn_warped_im:str,
    fn_transform_matrix:str,
    rng
    ):

    double_patch_size = np.int(2 * patch_size)

    for t in tqdm(range(num_patches)):

        # Get index for image patch on the fly
        if patch_type == 'training':
            i = np.int(np.ceil(rng.rand() * offset) * stride)
        else:
            i = patches.shape[0] - num_patches - 1 + t

        I = patches[i]
        I = np.transpose(I, [1, 2, 0])

        loop_success = False
        while not loop_success:
            try:
                tmp_translation = np.round((2 * rng.rand(2) - 2) * 8).astype('int')
                identity_matrix = np.eye(3)

                theta = (2 * rng.rand() - 1) * np.pi
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])

                tmp_scale = rng.uniform(low=0.85, high=1.15, size=2)
                scale_matrix = np.array([[tmp_scale[0], 0, 0],
                                        [0, tmp_scale[1], 0],
                                        [0, 0, 1]])

                tmp_shear = (2 * rng.rand(2) - 1) * 0.15
                shear_matrix = np.array([[1, 0, 0],
                                        [tmp_shear[0], 1, 0],
                                        [0, 0, 1]])

                # @ is matrix multiplication
                affine_matrix = shear_matrix @ scale_matrix @ rotation_matrix @ identity_matrix

                # transform patch
                J = warpPerspectiveNoCrop(I, affine_matrix)

                # Get center coordinates of original and transformed patch
                I_center_x = np.round(J.shape[1] / 2).astype('int')
                I_center_y = np.round(J.shape[0] / 2).astype('int')

                J_center_x = (np.round(J.shape[1] / 2) + tmp_translation[0]).astype('int')
                J_center_y = (np.round(J.shape[0] / 2) + tmp_translation[1]).astype('int')

                # Check boundary
                if J_center_x < patch_size:
                    J_center_x = patch_size
                    tmp_translation[0] = (patch_size - np.round(J.shape[1] / 2)).astype('int')

                if J_center_x + patch_size > J.shape[1]:
                    J_center_x = J.shape[1] - patch_size
                    tmp_translation[0] = (J.shape[1] - patch_size - np.round(J.shape[1] / 2)).astype('int')

                if J_center_y < patch_size:
                    J_center_y = patch_size
                    tmp_translation[1] = (patch_size - np.round(J.shape[0] / 2)).astype('int')

                if J_center_y + patch_size > J.shape[0]:
                    J_center_y = J.shape[0] - patch_size
                    tmp_translation[1] = (J.shape[0] - patch_size - np.round(J.shape[0] / 2)).astype('int')

                # standard patch, we add some rotation and shearing to standard patch
                crop_I = J[(I_center_y - patch_size) : (I_center_y + patch_size),
                        (I_center_x - patch_size) : (I_center_x + patch_size), :]

                # transformed patch
                crop_J = J[(J_center_y - patch_size) : (J_center_y + patch_size),
                        (J_center_x - patch_size) : (J_center_x + patch_size), :]

                # Write into arrays
                # gt transform
                transform_matrix = np.memmap(fn_transform_matrix,
                    dtype=np.float32,
                    mode='r+',
                    shape=(num_patches, feature_dim),
                    offset=t*feature_dim*np.dtype(np.float32).itemsize)

                transform_matrix[:1, :2] = np.array([
                    tmp_translation[0] / (double_patch_size / 3.0),
                    tmp_translation[1] / (double_patch_size / 3.0)]).astype(np.float32)

                im = np.memmap(fn_im,
                    dtype=np.float32,
                    mode='r+',
                    shape=(num_patches, double_patch_size, double_patch_size, num_channels),
                    offset=t*num_channels*double_patch_size*double_patch_size*np.dtype(np.uint8).itemsize)

                crop_I /= 255.0
                im[t, :double_patch_size, :double_patch_size, :num_channels] = crop_I

                warped_im = np.memmap(fn_warped_im,
                    dtype=np.float32,
                    mode='r+',
                    shape=(num_patches, double_patch_size, double_patch_size, num_channels),
                    offset=t*num_channels*double_patch_size*double_patch_size*np.dtype(np.uint8).itemsize)

                crop_J /= 255.0
                warped_im[t, :double_patch_size, :double_patch_size, :num_channels] = crop_J

                # If you came to this point, a patch has been created successfully.
                loop_success = True
            except Exception as e:
                # If invalid transformation is encountered try again.
                pass

def normalize_data(
    num_patches:int,
    fn_im:str,          # filepointer to patches memmeap
    fn_warped_im:str,   # filepointer to warped patches memmeap
    patch_size:int,
    num_channels:int=3
) -> None:

    double_patch_size = np.int(2 * patch_size)

    im = np.memmap(fn_im,
            dtype=np.float32,
            mode='r+',
            shape=(num_patches, double_patch_size, double_patch_size, num_channels))

    mean = np.mean(im)
    std = np.std(im)
    print('\tMean: {}\n\tStd: {}'.format(mean, std))

    print('\tNormalize patches...')
    for i in tqdm(range(num_patches)):
        im[i, :, :, :] = (im[i, :, :, :] - mean) / std

    del im

    im_warped = np.memmap(fn_warped_im,
        dtype=np.float32,
        mode='r+',
        shape=(num_patches, double_patch_size, double_patch_size, num_channels))

    print('\tNormalize warped patches...')
    for i in tqdm(range(num_patches)):
        im_warped[i, :, :, :] = (im_warped[i, :, :, :] - mean) / std

    del im_warped

parser = argparse.ArgumentParser()

parser.add_argument('--no_training',
    dest='no_training',
    action='store_true',
    help='Skip creation of trainings data.',
    default=False)

parser.add_argument('--no_test',
    dest='no_test',
    action='store_true',
    help='Skip creation of test data.',
    default=False)

parser.add_argument('--no_normalization',
    dest='no_normalization',
    action='store_true',
    help='Skip normalization of data.',
    default=False)

parser.add_argument('--num_test',
    type=int,
    help='Number of test patches to create. Default 128.',
    default=128)

parser.add_argument('--num_train',
    type=int,
    help='Number of training patches to create. Default 256000.',
    default=256000)

parser.add_argument('--seed',
    type=int,
    help='Seed number for random number generator. Default: 0',
    default=0)

if __name__ == '__main__':
    args = parser.parse_args()

    working_dir = './data/patch_set/'
    patch_dir = 'standard_patch'
    dataset_name =  'mexico_tilde_p24_Mexico'
    data_path = os.path.join(working_dir, patch_dir, dataset_name + '_patches.mat')

    # Load patches from matlab file into mat dictionary
    mat = {}
    f = h5py.File(data_path, 'r')
    for k, v in f.items():
        mat[k] = np.array(v).T
    print(mat.keys(), mat['patches'].shape) # (5879, 3, 51, 51)

    training_number = args.num_train
    test_number = args.num_test

    training_offset = 0
    offset = 5000
    stride = 1
    total_number = training_number + test_number
    feature_dim = 2
    num_channels = 3

    # random number generator
    rng = np.random.RandomState(seed=args.seed)

    patch_size = 32
    patch_size = np.int(patch_size / 2)
    double_patch_size = np.int(2 * patch_size)

    if not args.no_training:
        # Paths to memory maps for trainings data
        fn_train_im = os.path.join('memmaps', 'train', 'im')
        fn_train_warped_im = os.path.join('memmaps', 'train', 'warped_im')
        fn_train_transform_matrix = os.path.join('memmaps', 'train', 'transform_matrix')

        # You have to create the memmap-file once with w+ mode and open it later
        # wit h r+ mode, otherwise you would overwrite it everytime.
        im = np.memmap(fn_train_im,
            dtype=np.float32,
            mode='w+',
            shape=(training_number, double_patch_size, double_patch_size, num_channels))

        warped_im = np.memmap(fn_train_warped_im,
            dtype=np.float32,
            mode='w+',
            shape=(training_number, double_patch_size, double_patch_size, num_channels))

        transform_matrix = np.memmap(fn_train_transform_matrix,
            dtype=np.float32,
            mode='w+',
            shape=(training_number, feature_dim))

        print('\nCreate TRAINING data.')
        # Create trainings data
        create_training_pairs(
            training_number,
            'training',
            offset,
            training_offset,
            stride,
            feature_dim,
            num_channels,
            mat['patches'],
            patch_size,
            fn_train_im,
            fn_train_warped_im,
            fn_train_transform_matrix,
            rng)
        print('\nTRAINING data done.')

        # Normalize data
        if not args.no_normalization:
            print('\nNormalize TRAINING data.')
            normalize_data(
                training_number,
                fn_train_im,
                fn_train_warped_im,
                patch_size)
            print('\nNormalization done.')


        # Deletion flushes memory changes to disk before removing the object
        del im
        del warped_im
        del transform_matrix


    if not args.no_test:
        # Create test data
        #
        # Paths to memory maps for trainings data
        fn_test_im = os.path.join('memmaps', 'test', 'im')
        fn_test_warped_im = os.path.join('memmaps', 'test', 'warped_im')
        fn_test_transform_matrix = os.path.join('memmaps', 'test', 'transform_matrix')

        # You have to create the memmap-file once with w+ mode and open it later
        # wit h r+ mode, otherwise you would overwrite it everytime.
        im = np.memmap(fn_test_im,
            dtype=np.float32,
            mode='w+',
            shape=(test_number, double_patch_size, double_patch_size, num_channels))

        warped_im = np.memmap(fn_test_warped_im,
            dtype=np.float32,
            mode='w+',
            shape=(test_number, double_patch_size, double_patch_size, num_channels))

        transform_matrix = np.memmap(fn_test_transform_matrix,
            dtype=np.float32,
            mode='w+',
            shape=(test_number, feature_dim))

        print('\nCreate TEST data.')
        create_training_pairs(
            test_number,
            'test',
            offset,
            training_offset,
            stride,
            feature_dim,
            num_channels,
            mat['patches'],
            patch_size,
            fn_test_im,
            fn_test_warped_im,
            fn_test_transform_matrix,
            rng)
        print('\nTEST data done.')

             # Normalize data
        if not args.no_normalization:
            print('\nNormalize TRAINING data.')
            normalize_data(
                test_number,
                fn_test_im,
                fn_test_warped_im,
                patch_size)
            print('\nNormalization done.')

        # Deletion flushes memory changes to disk before removing the object
        del im
        del warped_im
        del transform_matrix


