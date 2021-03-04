import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import pickle
from utils_func import normalize


def window_select(data,timestep,window):
    if window ==0:
        return data[timestep : timestep + 1, :, :]
    max_len = data.shape[0]
    l = max(0,timestep-window)
    u = min(max_len,timestep+window)
    if l == 0:
        return (data[:2*window,:,:])
    elif u == max_len:
        return (data[-2*window:,:,:])
    else:
        return(data[l:u,:,:])


def get_subsample(touch, subsample):
    for x in range(0, touch.shape[1], subsample):
        for y in range(0, touch.shape[2], subsample):
            v = np.mean(touch[:, x:x+subsample, y:y+subsample], (1, 2))
            touch[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)

    return touch


class sample_data_diffTask(Dataset):
    def __init__(self, path, window, subsample):
        self.path = path
        self.files = os.listdir(self.path)
        self.subsample = subsample
        touch = np.empty((1,96,96))
        heatmap = np.empty((1,21,20,20,18))
        keypoint = np.empty((1,21,3))
        count = 0

        for f in self.files:
            count += 1
            print (f, count)
            data = pickle.load(open(self.path + f, "rb"))
            touch = np.append(touch, data[0], axis=0)
            heatmap = np.append(heatmap, data[1], axis=0)
            keypoint = np.append(keypoint, data[2], axis=0)


        self.data_in = [touch[1:,:,:], heatmap[1:,:,:,:,:], keypoint[1:,:,:]]
        self.window = window

    def __len__(self):
        # return self.length
        return self.data_in[0].shape[0]

    def __getitem__(self, idx):
        tactile = window_select(self.data_in[0],idx,self.window)
        heatmap = self.data_in[1][idx,:,:,:,:]
        keypoint = self.data_in[2][idx,:,:]
        tactile_frame = self.data_in[0][idx,:,:]

        if self.subsample > 1:
            tactile = get_subsample(tactile, self.subsample)

        return tactile, heatmap, keypoint, tactile_frame






