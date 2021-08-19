import numpy as np
from numpy import float32
import pickle
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from utils_func import tactile_reading, findFrame, normalize, softmax, normalize_with_range
from scipy.ndimage import gaussian_filter
from math import log10, floor
import math

BODY_25_color = np.array([[255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0], [255, 204, 0], [255, 255, 0], [204, 255, 0]
                         , [153, 255, 0], [102, 255, 0], [51, 255, 0], [0, 255, 0], [0, 255, 51], [0, 255, 102], [0,255,153]
                         , [0, 255, 204], [0, 255, 255], [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 53, 255], [0, 0, 255]
                         , [53, 0, 255], [102, 0, 255], [153, 0, 255], [204, 0, 255], [255, 0, 255]])

BODY_25_pairs = np.array([[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12],
                        [12, 13], [13, 14], [1, 0], [14, 19], [19, 20], [14, 21],
                         [11, 22], [22, 23], [11, 24]])
black = [0,0,0]


def round_to_1(data, sig):
    flag = np.where(data==0)
    data[flag] = int(0)
    c,x,y,z = np.where(data>0)
    for i in range(c.shape[0]):
        if data[c[i],x[i],y[i],z[i]] < 1e-2:
            data[c[i],x[i],y[i],z[i]] = 0
        else:
            data[c[i],x[i],y[i],z[i]] = round(data[c[i],x[i],y[i],z[i]], sig-int(floor(log10(abs(data[c[i],x[i],y[i],z[i]]))))-1)
    return data

def remove_keypoint_artifact(data,threshold):
    for q in range(3):
        frame = data[:,:,q]
        lower_flag = frame < threshold[q*2]
        upper_flag = frame > threshold[q*2 + 1]
        frame[lower_flag] = threshold[q*2]
        frame[upper_flag] = threshold[q*2 +1]
        data[:,:,q]=frame
    return data

def gaussian(dis, mu, sigma):
    return 1/(mu * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((dis - mu) /sigma)**2)

def plotKeypoint(data):
    fig = plt.figure()
    count = 0

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)

    # ax.set_xlim(-50,1500)
    # ax.set_ylim(-50,1500)
    # ax.set_zlim(-1800,0)
    ax.view_init(210,250)

    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    for i in range(BODY_25_pairs.shape[0]):
        index_1 = BODY_25_pairs[i, 0]
        index_2 = BODY_25_pairs[i, 1]
        if index_1 > 14:
            index_1 -= 4
        if index_2 > 14:
            index_2 -= 4

        xs_line = [xs[index_1],xs[index_2]]
        ys_line = [ys[index_1],ys[index_2]]
        zs_line = [zs[index_1],zs[index_2]]
        ax.plot3D(xs_line,ys_line,zs_line, color = BODY_25_color[i]/255.0)

    ax.scatter(xs, ys, zs, s=20, c=BODY_25_color[:21]/255.0)

    fig.canvas.draw()
    # plt.show()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    return img

def plot3Dheatmap(data, seperate):
    colors = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
              'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
              'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
              'BuGn', 'YlGn', 'Greys', 'Purples', 'Blues']

    if seperate:
        img_list = []
        for i in range(21):
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            ax.set_xlim(0,data.shape[1])
            ax.set_ylim(0,data.shape[2])
            ax.set_zlim(0,data.shape[3])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(210,250)

            frame = np.reshape(data[i,:,:,:],(data.shape[1],data.shape[2],data.shape[3]))
            flag = frame > 0
            x,y,z = np.where(frame>0)
            # print (frame[x,y,z])
            ax.scatter(x, y, z, c=frame[x,y,z]*255, cmap=colors[i])

            fig.canvas.draw()
            # plt.show()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img_frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]
            img_list.append(cv2.resize(img_frame, (320,240)))
            img = np.concatenate(img_list, axis=1)


    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.set_xlim(0,data.shape[1])
        ax.set_ylim(0,data.shape[2])
        ax.set_zlim(0,data.shape[3])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(210,250)

        for i in range(21):
            frame = np.reshape(data[i,:,:,:],(data.shape[1],data.shape[2],data.shape[3]))
            flag = frame > 0
            x,y,z = np.where(frame>0)
            # print (frame[x,y,z])
            ax.scatter(x, y, z, c=frame[x,y,z]*255, cmap=colors[i])

        fig.canvas.draw()
        # plt.show()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    return img


def heatmap_from_keypoint(keypoint_path, xyz_range, heatmap_size):
    '''
    Load triangulated, refined and transformed keypoint coordinate
    Build 3d voxel space
    Normalize keypoint in to 0-1 space, and save
    Generate heatmap with distance
    '''
    keypoint = pickle.load(open(keypoint_path, "rb"))
    x_range = xyz_range[0]
    y_range = xyz_range[1]
    z_range = xyz_range[2]
    size = heatmap_size

    resolution = [(x_range[1]-x_range[0])/size[0], (y_range[1]-y_range[0])/size[1], (z_range[1]-z_range[0])/size[2]]

    pos_y, pos_x, pos_z = np.meshgrid(
        np.linspace(0., 1., int(size[0])),
        np.linspace(0., 1., int(size[1])),
        np.linspace(0., 1., int(size[2])))

    print (pos_x.shape, pos_y.shape, pos_z.shape)

    heatmap = np.zeros((keypoint.shape[0],21,int(size[0]),int(size[1]),int(size[2])), dtype=float32)

    b = np.array([[x_range[0], y_range[0], z_range[0]]])
    threshold = [0,1,0,1,0,1]
    keypoint = normalize_with_range((keypoint - b)/resolution, max(size)-1, 0)
    keypoint = remove_keypoint_artifact(keypoint, threshold)

    print (np.amax(keypoint), np.amin(keypoint))

    for i in range(keypoint.shape[0]):
        if i % 500 == 0:
            print (i)
        # print (i)
        frame = np.reshape(keypoint[i,:,:], (21,3))

        for k in range(21):
            dis = np.sqrt((pos_x-frame[k,0])**2 + (pos_y-frame[k,1])**2 + (pos_z-frame[k,2])**2)
            g = gaussian(dis, 0.001, 1)
            # heatmap[i,k,:,:,:] = round_to_1(softmax(g),2)
            heatmap[i,k,:,:,:] = softmax(g) /0.25 #1:0.25; 0.5:0.8
            # print (np.amax(heatmap[i,k,:,:,:]))

    return keypoint, heatmap

'''enter folder names'''

path_list = ['./dataset/rec_2020-10-25_PM/',
             './dataset/rec_2020-10-25_YL/',
             './dataset/rec_2020-10-25_LS/']


shift_to_9tiles = True
shift = False
shift_range= []

for num, p in enumerate(path_list):
    path = p
    print (path)

    xyz_range = [[-100,1900],[-100,1900],[-1800,0]]
    size = [20, 20, 18] #define 3D space
    keypoint_path_list = [path + 'keypoint_transform.p']

    for num2, kp in enumerate(keypoint_path_list):
        keypoint_path = kp

        keypoint, heatmap = heatmap_from_keypoint(keypoint_path, xyz_range, size)
        print (np.amax(keypoint), np.amin(keypoint))

        pickle.dump(keypoint, open(keypoint_path + '_coord.p', "wb"))
        print ('Keypoint dumped:', keypoint.shape)

        l = 5000  #to prevent storage failures
        if heatmap.shape[0] <= l:
            pickle.dump(heatmap, open(keypoint_path+ '_heatmap3D.p', "wb"))
            print ('Heatmap3D dumped:', heatmap.shape)

        else:
            n = 0
            while heatmap.shape[0] - n > l:
                pickle.dump(heatmap[n:l + n, :, :, :, :], open(keypoint_path + '_heatmap3D_' + str(n) + '.p', "wb"))
                print ('Heatmap3D dumped:', n, heatmap.shape)
                n += l

            pickle.dump(heatmap[n:, :, :, :, :], open(keypoint_path + '_heatmap3D_' + str(n) + '.p', "wb"))
            print ('Heatmap3D dumped:', n, heatmap.shape)



    '''
    Check with visualization
    '''

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(path + 'heatmap.avi', fourcc, 10, (1280,480))
    print ('Video streaming')


    for i in range(1500,1600,10):
        #check heatmap viz
        print (i)

        img1 = plotKeypoint(np.reshape(keypoint[i,:,:], (21,3)))
        if shift_to_9tiles:
            img2 = plot3Dheatmap(round_to_1(np.reshape(heatmap[i,:,:,:,:]
                       ,(heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4])), 2), seperate=False)
        else:
            img2 = plot3Dheatmap(round_to_1(np.reshape(heatmap[i,:,:,:,:]
                       ,(heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4])), 2), seperate=False)
        # img3 = plot3Dheatmap(round_to_1(np.reshape(heatmap[i,:,:,:,:],(21, 16, 16, 16)), 2), seperate=True)

        # cv2.imshow('image',img1)
        # cv2.waitKey(0)
        # cv2.imshow('image',img2)
        # cv2.waitKey(0)
        # cv2.imshow('image',img3)
        # cv2.waitKey(0)

        image = np.concatenate((img1,img2), axis=1)
        out.write(image)




