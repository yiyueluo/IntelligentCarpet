import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from utils_func import tactile_reading, findFrame, normalize, readTs, tactile_to_3channel, normalize_with_range
from scipy.spatial.transform import Rotation as R

BODY_25_color = np.array([[255, 0, 0], [255, 51, 0], [255, 102, 0], [255, 153, 0], [255, 204, 0], [255, 255, 0], [204, 255, 0]
                         , [153, 255, 0], [102, 255, 0], [51, 255, 0], [0, 255, 0], [0, 255, 51], [0, 255, 102], [0,255,153]
                         , [0, 255, 204], [0, 255, 255], [0, 204, 255], [0, 153, 255], [0, 102, 255], [0, 53, 255], [0, 0, 255]
                         , [53, 0, 255], [102, 0, 255], [153, 0, 255], [204, 0, 255], [255, 0, 255]])

BODY_25_pairs = np.array([[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12],
                        [12, 13], [13, 14], [1, 0], [14, 19], [19, 20], [14, 21],
                         [11, 22], [22, 23], [11, 24]])
black = (0,0,0)
feet_pairs = np.array([[0,3],[1,2]])
torso_pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [7, 8], [4, 9]])
leg_pairs = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [5, 6], [2, 7]])

color = np.array([255,255,255])

tile_pos = np.array([[2.0797312,  -10.861213,   -19.048914 ],
                     [1830.4335,  -47.5771,     -13.48735  ],
                     [1831.6855,  1820.6862,    -18.654463 ],
                     [20.110088,  1853.9688,    -21.118084 ]])

def remove_samll(data):
    flag = np.where(data==0)
    data[flag] = int(0)
    c,x,y,z = np.where(data>0)
    for i in range(c.shape[0]):
        if data[c[i],x[i],y[i],z[i]] < 0.05:
            data[c[i],x[i],y[i],z[i]] = 0
    return data

def rotate(touch, heatmap, keypoint, degree):
    r = R.from_euler('z', degree, degrees=True)
    r = r.as_matrix()
    b = np.array([0.5, 0.5, 0 ])
    heatmap_r = np.copy(heatmap)
    keypoint_r = np.copy(keypoint)
    touch_r = np.copy(touch)

    for frame in range(keypoint.shape[0]):
        touch_r[frame,:,:] = np.rot90(np.reshape(touch[frame,:,:], (touch.shape[1],touch.shape[2])), k=degree/90, axes=(1,0))

        for i in range(keypoint.shape[1]):
            heatmap_r[frame,i,:,:,:] = np.rot90(np.reshape(heatmap[frame,i,:,:,:]
                                       ,(heatmap.shape[2],heatmap.shape[3],heatmap.shape[4])), degree/90, (0,1))
            keypoint_r[frame,i,:] = np.dot(r,(np.reshape(keypoint[frame,i,:], (3)) - b)) + b

    return touch_r, heatmap_r, keypoint_r


def plot_touch(touch_frame, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(touch_frame, cmap='viridis')
    # ax.set_cmap('viridis')
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def plot_touch2(touch_frame):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos_x, pos_y = np.meshgrid(
    np.linspace(0, 96, 96, endpoint=False),
    np.linspace(0, 96, 96, endpoint=False))
    touch_frame = touch_frame * 15
    ax.scatter(pos_x, pos_y, linewidths=.2, edgecolors='k', c='k', s =touch_frame)
    ax.set_aspect('equal')
    plt.axis('off')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    return img

def plotKeypoint(data, tactile, scale, tile_coord, tactile_frame, topVeiw, keypoint):

    b = [-100,-100,-1800]
    resolution = 100

    fig = plt.figure()
    count = 0

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-10,190)
    ax.set_ylim(-10,190)
    ax.set_zlim(180,0)

    plt.xticks([0,50,100,150])
    plt.yticks([0,50,100,150])
    ax.set_zticks([0,50,100,150])

    if topVeiw:
        ax.view_init(80,250)

    else:
        ax.view_init(210,230)

    xs = tile_coord[:, 0]/10
    ys = tile_coord[:, 1]/10
    zs = tile_coord[:, 2]/10


    if tactile:
        x_range = np.abs(round(xs[1])-round(xs[0]))
        y_range = np.abs(round(ys[2])-round(ys[1]))
        # tactile_frame = cv2.resize(tactile_frame, (x_range, y_range))
        X, Y = np.meshgrid(np.linspace(round(xs[0]),round(xs[1]),96), np.linspace(round(ys[1]),round(ys[2]),96))
        Z = np.copy(X)
        Z[:,:] = zs[0]


        ax.scatter(X, Y, linewidths=.2, edgecolors='k', c='k', s=tactile_frame*5, zorder=1)

        # if topVeiw:
        #     ax.contourf(X, Y, tactile_frame, 0, zdir='z', offset=20, cmap='viridis')
        # else:
        #     # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.viridis(tactile_frame), shade=False)
        #     ax.contourf(X, Y, tactile_frame, 0, zdir='z', offset=round(zs[0]), cmap='binary', zorder=1)

    if keypoint:
        data = data * scale
        data = data * resolution + b

        xs = data[:, 0]/10
        ys = data[:, 1]/10
        zs = -data[:, 2]/10


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
            ax.plot(xs_line,ys_line,zs_line, color = BODY_25_color[i]/255.0, linewidth=5, zorder=10)

        ax.scatter(xs, ys, zs, s=50, c=BODY_25_color[:21]/255.0, zorder=10)

    fig.canvas.draw()
    # plt.show()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    return img

def plot3Dheatmap(data):
    colors = ['Reds','PuRd', 'Oranges', 'YlOrRd', 'YlOrBr','Greens','BuGn', 'YlGn',
              'PuRd', 'GnBu', 'YlGnBu', 'Blues', 'YlOrRd', 'YlOrBr', 'Greens', 'GnBu', 'YlGnBu', 'Blues','Greens','BuGn', 'YlGn']

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlim(-1,19)
    ax.set_ylim(-1,19)
    ax.set_zlim(0,18)

    plt.xticks([0,5,10,15])
    plt.yticks([0,5,10,15])
    ax.set_zticks([0,5,10,15])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(210,230)

    # for i in range(data.shape[0]):
    for i in range(21):
        frame = np.reshape(data[i,:,:,:],(20,20,18))
        flag = frame > 0
        x,y,z = np.where(frame>0)
        ax.scatter(x-1, y-1, z, c=frame[x,y,z]*255, cmap=colors[i])

    fig.canvas.draw()
    # plt.show()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    return img

def generateImage(data, path, c, base, tile_coord=tile_pos):

    heatmap_GT = data[0]
    heatmap_pred = data[1]
    keypoint_GT = data[2]
    keypoint_pred = data[3]
    touch = data[4]

    t, heatmap_GT, keypoint_GT = rotate(touch, heatmap_GT, keypoint_GT, 90)
    touch, heatmap_pred, keypoint_pred = rotate(touch, heatmap_pred, keypoint_pred, 90)

    # print (heatmap_GT.shape, heatmap_pred.shape)

    '''save image'''

    for i in range(touch.shape[0]): #keypoint_GT.shape[0]):
        print (i)
        tactile_frame = np.reshape(touch[i,:,:],(96,96))

        # plot_touch(tactile_frame,  path + 'tactile%06d.jpg' % (base+c*32+i))
        img = plot_touch2(tactile_frame)
        cv2.imwrite(path + 'tactile%06d.jpg' % (base+c*32+i), img)


        img1 = plot3Dheatmap(remove_samll(np.reshape(heatmap_GT[i,:,:,:,:],(21,20,20,18))))
        img2 = plot3Dheatmap(remove_samll(np.reshape(heatmap_pred[i,:,:,:,:],(21,20,20,18))))

        cv2.imwrite(path + 'heatmap_gt%06d.jpg' % (base+c*32+i), img1)
        cv2.imwrite(path + 'heatmap_pred%06d.jpg' % (base+c*32+i), img2)


        img3 = plotKeypoint(np.reshape(keypoint_GT[i,:,:],(21,3)), scale=19, tactile=True
                                       , tile_coord=tile_coord, tactile_frame=tactile_frame
                                       , topVeiw=False, keypoint=False)
        img5 = plotKeypoint(np.reshape(keypoint_GT[i,:,:],(21,3)), scale=19, tactile=True
                                       , tile_coord=tile_coord, tactile_frame=tactile_frame
                                       , topVeiw=False, keypoint=True)
        img6 = plotKeypoint(np.reshape(keypoint_pred[i,:,:],(21,3)), scale=19, tactile=True
                                       , tile_coord=tile_coord, tactile_frame=tactile_frame
                                       , topVeiw=False, keypoint=True)


        cv2.imwrite(path + '3Dtactile%06d.jpg' % (base+c*32+i), img3)
        cv2.imwrite(path + 'gt%06d.jpg' % (base+c*32+i), img5)
        cv2.imwrite(path + 'pred%06d.jpg' % (base+c*32+i), img6)


        # data = [np.reshape(keypoint_GT[i,:,:],(21,3)), np.reshape(keypoint_pred[i,:,:],(21,3))]
        # img5 = cv2.resize(plotKeypoint(data, scale=19, tactile=True, tile_coord=tile_coord, tactile_frame=tactile_frame
        #                                , topVeiw=False, GT_pred_compare=True),(640,480))

        # img5 = cv2.resize(plotKeypoint(np.reshape(keypoint_GT[i,:,:],(21,3)), scale=100, tactile=False
        #                                , tile_coord=tile_coord, tactile_frame=tactile_frame, topVeiw=False),(320,240))
        # img6 = cv2.resize(plotKeypoint(np.reshape(keypoint_pred[i,:,:],(21,3)), scale=100, tactile=False
        #                                , tile_coord=tile_coord, tactile_frame=tactile_frame, topVeiw=False),(320,240))
        # img7 = cv2.resize(plotKeypoint(np.reshape(keypoint_GT[i,:,:],(21,3)), scale=100, tactile=True
        #                                , tile_coord=tile_coord, tactile_frame=tactile_frame, topVeiw=True),(320,240))
        # img8 = cv2.resize(plotKeypoint(np.reshape(keypoint_pred[i,:,:],(21,3)), scale=100, tactile=True
        #                                , tile_coord=tile_coord, tactile_frame=tactile_frame, topVeiw=True),(320,240))


        # print (i)

