import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from utils_func import tactile_reading, findFrame, normalize, readTs, tactile_to_3channel, normalize_with_range

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
        if data[c[i],x[i],y[i],z[i]] < 1e-2:
            data[c[i],x[i],y[i],z[i]] = 0
    return data

def plotKeypoint(data, tactile, scale, tile_coord, tactile_frame, topVeiw, GT_pred_compare):

    b = [-100,-100,-1800]
    resolution = 100

    fig = plt.figure()
    count = 0

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-100,1900)
    ax.set_ylim(-100,1900)
    ax.set_zlim(-1800,0)

    if topVeiw:
        ax.view_init(80,250)

    else:
        ax.view_init(210,250)

    # tile_coord = normalize_with_range((tile_coord - b)/resolution, 15, 0)*scale

    xs = tile_coord[:, 0]
    ys = tile_coord[:, 1]
    zs = tile_coord[:, 2]
    ax.plot3D([xs[0],xs[1],xs[2],xs[3],xs[0]],[ys[0],ys[1],ys[2],ys[3],ys[0]]
              ,[zs[0],zs[1],zs[2],zs[3],zs[0]], color=black)
    # ax.fill([xs[0],xs[1],xs[2],xs[3],xs[0]],[ys[0],ys[1],ys[2],ys[3],ys[0]]
    #           ,[zs[0],zs[1],zs[2],zs[3],zs[0]], color=black)
    ax.scatter(xs, ys, zs, s=20,color=black)

    if tactile:
        x_range = np.abs(round(xs[1])-round(xs[0]))
        y_range = np.abs(round(ys[2])-round(ys[1]))
        tactile_frame = cv2.resize(tactile_frame, (x_range, y_range))
        X, Y = np.meshgrid(np.linspace(round(xs[0]),round(xs[1]),x_range), np.linspace(round(ys[1]),round(ys[2]),y_range))
        # cset = ax.contourf(X, Y, tactile_frame, 100, zdir='z', offset=round(zs[0]), cmap='binary')

        if topVeiw:
            cset = ax.contourf(X, Y, tactile_frame, 0, zdir='z', offset=20, cmap='binary')
        else:
            cset = ax.contourf(X, Y, tactile_frame, 0, zdir='z', offset=round(zs[0]), cmap='binary')

    if GT_pred_compare:
        for n in range(len(data)):

            data[n] = data[n] * scale
            data[n] = data[n] * resolution + b

            xs = data[n][:, 0]
            ys = data[n][:, 1]
            zs = data[n][:, 2]

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

                if n == 0:
                    ax.plot3D(xs_line,ys_line,zs_line, color = black)
                else:
                    ax.plot3D(xs_line,ys_line,zs_line, color = BODY_25_color[i]/255.0)

            if n == 0:
                ax.scatter(xs, ys, zs, s=50, color=black)
            else:
                ax.scatter(xs, ys, zs, s=50, c=BODY_25_color[:21]/255.0)


    else:
        data = data * scale
        data = data * resolution + b

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

        ax.scatter(xs, ys, zs, s=50, c=BODY_25_color[:21]/255.0)

    fig.canvas.draw()
    # plt.show()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    return img

def plot3Dheatmap(data):
    colors = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
          'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
          'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
          'BuGn', 'YlGn', 'Greys', 'Purples', 'Blues']

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlim(0,20)
    ax.set_ylim(0,20)
    ax.set_zlim(0,18)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(210,250)

    # for i in range(data.shape[0]):
    for i in range(21):
        frame = np.reshape(data[i,:,:,:],(20,20,18))
        flag = frame > 0
        x,y,z = np.where(frame>0)
        ax.scatter(x, y, z, c=frame[x,y,z]*255, cmap=colors[i])

    fig.canvas.draw()
    # plt.show()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

    return img

def generateVideo(data, path, heatmap, tile_coord=tile_pos):

    heatmap_GT = data[0]
    heatmap_pred = data[1]
    keypoint_GT = data[2]
    keypoint_pred = data[3]
    touch = data[4]

    # print (heatmap_GT.shape, heatmap_pred.shape)

    '''save video'''
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if heatmap:
        out = cv2.VideoWriter(path + '.avi', fourcc, 10, (1280,480))
    else:
        out = cv2.VideoWriter(path + '.avi', fourcc, 10, (960,480))
    print ('Video streaming')

    for i in range(keypoint_GT.shape[0]):
        print (i)
        tactile_frame = np.reshape(touch[i,:,:],(96,96))

        if heatmap:
            img1 = cv2.resize(plot3Dheatmap(remove_samll(np.reshape(heatmap_GT[i,:,:,:,:],(21,20,20,18)))),(320,240))
            img2 = cv2.resize(plot3Dheatmap(remove_samll(np.reshape(heatmap_pred[i,:,:,:,:],(21,20,20,18)))),(320,240))


        img3 = cv2.resize(plotKeypoint(np.reshape(keypoint_GT[i,:,:],(21,3)), scale=19, tactile=False
                                       , tile_coord=tile_coord, tactile_frame=tactile_frame
                                       , topVeiw=False, GT_pred_compare=False),(320,240))
        img4 = cv2.resize(plotKeypoint(np.reshape(keypoint_pred[i,:,:],(21,3)), scale=19, tactile=False
                                       , tile_coord=tile_coord, tactile_frame=tactile_frame
                                       , topVeiw=False, GT_pred_compare=False),(320,240))

        # cv2.imwrite('./recordings/img/gt%06d.jpg' % i, img3)
        # cv2.imwrite('./recordings/img/tactile%06d.jpg' % i, (1-tactile_frame) * 255)
        # cv2.imwrite('./recordings/img/pred%06d.jpg' % i, img4)

        data = [np.reshape(keypoint_GT[i,:,:],(21,3)), np.reshape(keypoint_pred[i,:,:],(21,3))]
        img5 = cv2.resize(plotKeypoint(data, scale=19, tactile=True, tile_coord=tile_coord, tactile_frame=tactile_frame
                                       , topVeiw=False, GT_pred_compare=True),(640,480))

        # img5 = cv2.resize(plotKeypoint(np.reshape(keypoint_GT[i,:,:],(21,3)), scale=100, tactile=False
        #                                , tile_coord=tile_coord, tactile_frame=tactile_frame, topVeiw=False),(320,240))
        # img6 = cv2.resize(plotKeypoint(np.reshape(keypoint_pred[i,:,:],(21,3)), scale=100, tactile=False
        #                                , tile_coord=tile_coord, tactile_frame=tactile_frame, topVeiw=False),(320,240))
        # img7 = cv2.resize(plotKeypoint(np.reshape(keypoint_GT[i,:,:],(21,3)), scale=100, tactile=True
        #                                , tile_coord=tile_coord, tactile_frame=tactile_frame, topVeiw=True),(320,240))
        # img8 = cv2.resize(plotKeypoint(np.reshape(keypoint_pred[i,:,:],(21,3)), scale=100, tactile=True
        #                                , tile_coord=tile_coord, tactile_frame=tactile_frame, topVeiw=True),(320,240))

        image2 = np.concatenate((img3, img4), axis=0)
        # image3 = np.concatenate((img5, img6), axis=0)
        # image5 = np.concatenate((img7, img8), axis=0)

        if heatmap:
            image1 = np.concatenate((img1, img2), axis=0)
            image4 = np.concatenate((image1, image2), axis=1)
            image = np.concatenate((image4, img5), axis=1)
            # image6 = np.concatenate((image4, image3), axis=1)
            # image = np.concatenate((image6, image5), axis=1)
        else:
            image = np.concatenate((image2, img5), axis=1)


        # print (image.shape)
        # print (image1.shape, image2.shape)

        # cv2.imshow('image', image)
        # cv2.waitKey(1)

        out.write(image)
    exit(0)

