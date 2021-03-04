import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import os
import random
import h5py
import matplotlib.patches as patches
import json
import scipy.io
import _pickle as cPickle
import bz2

def findFrame(ts_target, ts_set):
    idx = (np.abs(ts_target-ts_set)).argmin()
    return idx

def readTs(path =''):
    with open(path,'r') as keypoint_tsFile:
        ts = np.array([float(line.rstrip('\n')) for line in keypoint_tsFile])
    return ts

def tactile_reading(path=''):
    f = h5py.File(path, 'r')
    fc = f['frame_count'][0]
    touch_ts = np.array(f['ts'][:fc])
    touch_data = np.array(f['pressure'][:fc]).astype(np.float32)
    return touch_data, fc, touch_ts

def webcam_reading(path=''):
    cam = cv2.VideoCapture(path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    fc = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = fc/fps
    ret_val, image = cam.read()
    return image

def plotImg(path):
    img = cv2.imread(path)
    plt.imshow(img)
    plt.show()
    return None

def draw_channel(color, heatmap, keypoint_num, size):
    color = np.asanyarray(color)
    heatmap_viz = heatmap[:, :, :, None] * color[:keypoint_num, None, None, :]
    heatmap_viz = np.sum(heatmap_viz, 0)
    heatmap_viz = cv2.resize(heatmap_viz,(size[0],size[1]))
    return heatmap_viz

def draw_keypoint2D(pairs, colors, data, keypoint_count, size):
    img = np.zeros([720,1280,3],dtype=np.uint8)
    img.fill(255)
    coordinate = []
    for k in range(keypoint_count):
        coordinate.append((round(data[2*k]),round(data[2*k+1])))
        cv2.circle(img, coordinate[k], 1, colors[k], thickness=1, lineType=1, shift=0)

    for pair_order, pair in enumerate(pairs):
        cv2.line(img, coordinate[pair[0]], coordinate[pair[1]], colors[pair_order], 2)

    img = cv2.resize(img,(size[0],size[1]))
    return img

def normalize(data):
    data = (data-np.amin(data))/(np.amax(data)-np.amin(data))
    return data

def normalize_with_range(data, max, min):
    data = (data-min)/(max-min)
    return data

def softmax(x):
    output = np.exp(x) / np.sum(np.exp(x))
    return output

def sigmoid(x):
    output = 1/ 1 + np.exp(-x)
    return output

def outputImage(inputVideo, outputPath):
    cam = cv2.VideoCapture(inputVideo)
    fc = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print ('fc:', fc)
    count = 0
    while count < fc:
        ret_val, image = cam.read()
        if ret_val:
            cv2.imwrite(outputPath + 'frame%d.jpg' % count, image)
            count += 1
        else:
            break
    print ('Finished:', count)

def tactile_to_3channel(tactileFrame):
    color = np.array([255,255,255])
    image = tactileFrame[:, :, None] * color[None, None, :]
    image = image.clip(0, 255).astype(np.uint8)
    return image

