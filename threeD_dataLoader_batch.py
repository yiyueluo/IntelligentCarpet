import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import pickle
from utils_func import normalize
import time


def window_select(log, path, f ,idx, window):
    if window == 0:
        d = pickle.load(open(path + '/' + str(idx) + '.p', "rb"))

        return np.reshape(d[0], (1,96,96)), d[1], d[2], np.reshape(d[0], (1,96,96))


    max_len = log[f+1]
    min_len = log[f]
    l = max([min_len, idx-window])
    u = min([max_len, idx+window])

    dh = pickle.load(open(path + '/' + str(idx) + '.p', "rb"))
    heatmap = dh[1]
    keypoint = dh[2]
    tactile_frame = np.reshape(dh[0], (1,96,96))

    tactile = np.empty((2*window, 96, 96))

    if l == min_len:
        for i in range(min_len, min_len+2*window):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i-min_len,:,:] = d[0]

        return tactile, heatmap, keypoint, tactile_frame

    elif u == max_len:
        for i in range(max_len-2*window, max_len):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i-(max_len-2*window),:,:] = d[0]

        return tactile, heatmap, keypoint, tactile_frame

    else:
        for i in range(l, u):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i-l,:,:] = d[0]

        return tactile, heatmap, keypoint, tactile_frame


def get_subsample(touch, subsample):
    for x in range(0, touch.shape[1], subsample):
        for y in range(0, touch.shape[2], subsample):
            v = np.mean(touch[:, x:x+subsample, y:y+subsample], (1, 2))
            touch[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)

    return touch



class sample_data(Dataset):
    def __init__(self, path, window, mask, subsample):
        self.mask = mask
        self.path = path
        self.window = window
        self.subsample = subsample
        self.log = pickle.load(open(self.path + 'log.p', "rb"))

    def __len__(self):
        # return self.length
        if self.mask != []:
            # print (self.log[-1], np.amin(self.mask))
            return self.log[-1] + self.mask[-1]
        else:
            return self.log[-1]

    def __getitem__(self, idx):

        if self.mask != []:
            f = np.where((self.log + self.mask)<=idx)[0][-1]
            local_path = os.path.join(self.path, str(self.log[f]))
            tactile, heatmap, keypoint, tactile_frame = window_select(self.log, local_path, f, idx-self.mask[f], self.window)
        else:
            f = np.where(self.log<=idx)[0][-1]
            local_path = os.path.join(self.path, str(self.log[f]))
            tactile, heatmap, keypoint, tactile_frame = window_select(self.log, local_path, f, idx, self.window)

        if self.subsample > 1:
            tactile = get_subsample(tactile, self.subsample)

        return tactile, heatmap, keypoint, tactile_frame, idx
