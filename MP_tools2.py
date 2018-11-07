import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import dpcore
import os
from munkres import Munkres # Optimization Algorithm(Hungarian Algo) / find the global minimum

def load_data(input_dir, dataset_dir):

    ##### Import json #####
    with open(input_dir) as f:
        data = json.load(f)

    with open(dataset_dir) as l:
        dataLims = json.load(l)

    return data, dataLims


def Create3dPoints(init_frame, last_frame, data, model_name):
    points = []
    for i in range(init_frame, last_frame):
        strnum = str(i)
        frames = []
        for item in data['landmarks'][strnum][model_name]:
            frames.extend(item)

        points.append(frames)

    p3d = np.array(points)

    return p3d


def GaussFilter3dPoints(p3d, sigma, t):
    f_gauss = gaussian_filter1d(p3d[:, 0], sigma=sigma, truncate=t)

    for x in range(1, p3d.shape[1]):
        f_gauss = np.column_stack((f_gauss, gaussian_filter1d(p3d[:, x], sigma=sigma, truncate=t)))

    # Feature Vector with filtered coordinates [1001 x 126]
    p3d_gauss = f_gauss

    # print(p3d.shape)
    # print(p3d_gauss.shape)

    return p3d_gauss


def MovPoseDescriptor(p3d_gauss, StartFrame):
    # f32 = np.dtype(np.float32)
    sz_p3d = p3d_gauss.shape
    pt0 = []
    pt1 = []
    pt2 = []
    ptm1 = []
    ptm2 = []
    for fr in range(StartFrame, sz_p3d[0] - StartFrame):
        Pt0 = []
        Pt1 = []
        Pt2 = []
        Ptm1 = []
        Ptm2 = []
        for cl in range(0, sz_p3d[1]):
            Pt0.extend([p3d_gauss[fr, cl]])
            Pt1.extend([p3d_gauss[fr + 1, cl]])
            Pt2.extend([p3d_gauss[fr + 2, cl]])
            Ptm1.extend([p3d_gauss[fr - 1, cl]])
            Ptm2.extend([p3d_gauss[fr - 2, cl]])

        pt0.append(Pt0)
        pt1.append(Pt1)
        pt2.append(Pt2)
        ptm1.append(Ptm1)
        ptm2.append(Ptm2)

    Pt0 = np.array(pt0)
    Pt1 = np.array(pt1)
    Pt2 = np.array(pt2)
    Ptm1 = np.array(ptm1)
    Ptm2 = np.array(ptm2)

    ## Acc / Vec

    vec = Pt1 - Ptm1
    acc = Pt2 + Ptm2 - 2 * Pt0
    # print(vec.shape)
    # print(acc.shape)

    ## magnitude of vel /  and magnitude of acc
    # magvec = np.ndarray((vec.shape[0], 15), dtype=float)
    # magacc = np.ndarray((vec.shape[0], 15), dtype=float)
    # magvec = np.zeros((vec.shape[0], 15), dtype=float)
    # magacc = np.zeros((vec.shape[0], 15), dtype=float)
    magvec = np.empty((vec.shape[0], 15), np.dtype(np.float32))
    magacc = np.empty((vec.shape[0], 15), np.dtype(np.float32))

    for xf in range(0, vec.shape[0]):
        indx = 0
        for xp in range(0, 43, 3):
            pointv = vec[xf, xp:xp + 3]
            pointa = acc[xf, xp:xp + 3]
            magvec[xf][indx] = np.linalg.norm(pointv)
            magacc[xf][indx] = np.linalg.norm(pointa)
            indx = indx + 1

    ## Feature Vector
    # f_v = np.concatenate((Pt0, vec, acc), axis=1)
    f_v = np.concatenate((Pt0, magvec, magacc), axis=1)
    z = np.copy(f_v[f_v.shape[0] - 1, :])
    z = np.matlib.repmat(z, 4, 1)
    feat_vec = np.vstack((f_v, z))

    print(feat_vec.shape)
    return feat_vec, vec, acc


def smoothPlot(p3d, p3dsmooth):
        # Gaussian Smoothing - Plot ##
        fig1, ax1 = plt.subplots()

        ax1.plot(p3d[:, 3], label='Before Smooth')
        ax1.plot(p3dsmooth[:, 3], label='After Smooth')

        plt.xlabel('frames')
        plt.ylabel('X coord')
        plt.title('X coordinate before &\nafter Gaussian smoothing')
        plt.legend()
        plt.show(ax1)
        plt.close('all')

def dtwC(dist_mat, penalty):
    p, q, C, phi = dpcore.dp(dist_mat, penalty=penalty)
    return p, q, C, phi

def DistMatPlot(f_v, path, q=None, p=None,dtwscore=None, name=None, flag=None, save_flag=None):

    if save_flag == 1:

        goal_dir = os.path.join(path)
        fig, ax = plt.subplots()
        cax = ax.matshow(f_v, interpolation='None')

        ax.grid(True)
        plt.xlabel('frames')
        plt.ylabel('frames')

        if flag == 'similarity':
            fig.colorbar(cax)
            my_file = name + '_sim_mat'
            plt.title('Self Similarity Matrix\n Moving Pose Descriptor')
        elif flag == 'compare':
            fig.colorbar(cax)
            my_file = name + '_comp_mat'
            plt.title('Distance Matrix\n Comparison ' + name)
        elif flag == 'DTW':
            # ax.imshow(f_v, interpolation='nearest', cmap='binary')
            ax.hold(True)
            ax.plot(q, p, '-r')
            ax.hold(False)
            ax.autoscale(enable=True, axis='both', tight=True)
            my_file = name + '_dtw_path'
            plt.title('MP - DTW Score: '+str(dtwscore))

        plt.close('all')
        # print(goal_dir + my_file)
        fig.savefig(goal_dir + my_file, bbox_inches='tight')


    else:
        print('Passing Plot...')
        pass

def Optimize(score):

    matrix = score.copy()
    murk = Munkres()
    matrix[5][5]=20000
    indexes = murk.compute(matrix)
    # print (matrix, 'Lowest cost through this matrix:')
    total = 0

    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
        print 'total cost: %d' % total

    return indexes