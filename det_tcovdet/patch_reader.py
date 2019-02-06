import collections
import os
import scipy.io as sio
import tensorflow as tf
import numpy as np
import random
import cv2
from tqdm import tqdm

from scipy.spatial import distance

class SiameseDataSet(object):
    def __init__(self,
                 base_dir,
                 num_patches,
                 test = False
                 ):

        self.test = test
        self.n  = 128
        self._base_dir = base_dir

        self.PATCH_SIZE = 32
        self.num_patches = num_patches
        self.feature_dim = 2
        # the loaded patches
        self._data = dict()

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_train_patch(self):
        return self._num_train_patch

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index(self):
        return self._index

    def _get_data(self):
        return self._data

    def _get_patches(self):
        return self._get_data()['patches']

    def _get_patches_transformed(self):
        return self._get_data()['patches_transformed']

    def _get_matrix(self):
        return self._get_data()['labels']


    def load_by_name(self, name, patch_size=32, num_channels=3, debug=True):
        print('\n----\nload_by_name: \nname: ', name, '\npatch_size: ', 32, '\nnum_channels: ', num_channels, '\ndebug: ', debug)
        patches, patches_transformed, transform_matrix = self._load_patches(self._base_dir, name, patch_size, num_channels)

        # load the labels
        self._data['patches'] = patches
        self._data['patches_transformed'] = patches_transformed
        self._data['labels']  =  transform_matrix

        if debug:
            print('-- Dataset loaded:    %s' % name)
            print('-- Number of patches: %s' % len(self._data['patches']))
            print('-- Number of labels:  %s' % len(self._data['labels']))

    def _load_patches(self, base_dir, name, patch_size, num_channels):
        fp_patches = os.path.join(base_dir, 'im')
        fp_patches_transformed = os.path.join(base_dir, 'warped_im')
        fp_labels = os.path.join(base_dir, 'transform_matrix')

        patches_all = np.memmap(
            fp_patches,
            dtype=np.float32,
            mode='r+',
            shape=(self.num_patches, self.PATCH_SIZE, self.PATCH_SIZE, num_channels))

        patches_transformed = np.memmap(
            fp_patches_transformed,
            dtype=np.float32,
            mode='r+',
            shape=(self.num_patches, self.PATCH_SIZE, self.PATCH_SIZE, num_channels))

        transform_matrix = np.memmap(
            fp_labels,
            dtype=np.float32,
            mode='r+',
            shape=(self.num_patches, self.feature_dim))

        print('\n----\n_load_patches\nbase_dir: ', base_dir, '\nname: ', name, '\npatch_size: ', patch_size, '\nnum_channels: ', num_channels)
        print(patches_all.shape)
        print(patches_transformed.shape)
        print(transform_matrix.shape)

        self._num_train_patch = transform_matrix.shape[0]

        return patches_all, patches_transformed, transform_matrix

    def _create_indices(self, labels):
        old = labels[0]
        indices = dict()
        indices[old] = 0
        for x in range(len(labels) - 1):
            new = labels[x + 1]
            if old != new:
                indices[new] = x + 1
            old = new
        return indices

    def _compute_mean_and_std(self, patches):
        assert len(patches) > 0, 'Patches list is empty!'
        # compute the mean
        mean = np.mean(patches)
        std = np.std(patches)
        return mean, std

    def normalize_data(self, mean, std):
        #pbar = tqdm(self._data['patches'])
        pbar = tqdm(range(self._data['patches'].shape[0]))
        for i in pbar:
            pbar.set_description('Normalizing data')
            self._data['patches'][i, :, :, :] = (self._data['patches'][i, :, :, :]  - mean) / std
            self._data['patches_transformed'][i, :, :, :] = (self._data['patches_transformed'][i, :, :, :] - mean) / std

    def generate_stats(self):
        #print('-- Computing dataset mean: %s ...' % name)
        # compute the mean and std of all patches
        patches = self._get_patches()
        mean, std = self._compute_mean_and_std(patches)
        print('-- Computing dataset mean:  ... OK')
        print('-- Mean: %s' % mean)
        print('-- Std : %s' % std)
        return mean, std

    def prune(self, min=2):
        labels = self._get_labels()
        ids, labels = self._prune(labels, min)
        return ids, labels

    def _prune(self, labels, min):
        # count the number of labels
        c = collections.Counter(labels)
        # create a list with globals indices
        ids = range(len(labels))
        # remove ocurrences
        ids, labels = self._rename_and_prune(labels, ids, c, min)
        return np.asarray(ids), np.asarray(labels)

    def _rename_and_prune(self, labels, ids, c, min):
        count, x = 0, 0
        labels_new, ids_new = [[] for _ in range(2)]
        while x < len(labels):
            num = c[labels[x]]
            if num >= min:
                for i in range(num):
                    labels_new.append(count)
                    ids_new.append(ids[x + i])
                count += 1
            x += num
        return ids_new, labels_new

    def generate_index(self):
        # retrieve loaded patches and labels
        labels = self._get_matrix()

        index = np.arange(labels.shape[0])
        np.random.shuffle(index)

        self._index = index

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
            self._epochs_completed += 1
        # Go to the next epoch
        if start + batch_size > self._num_train_patch:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = 0
            #temp_index = np.arange(self._num_train_patch)
            np.random.shuffle(self._index)
            #self._index = temp_index

        self._index_in_epoch += batch_size
        end = self._index_in_epoch

        return self._index[start:end]

    def iter_select(self,index):
        # retrieve loaded patches and labels
        index = np.asarray(index)
        np.random.shuffle(index)
        self._num_train_patch = index.shape[0]
        self._index = index
