import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os

def load_data(name):

    dataset_dir = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/" + name + ".json"
    input_dir = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/rs/Human_tracking/" \
                + name + "_results_ldm.json"

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

    print(p3d.shape)
    print(p3d_gauss.shape)

    return p3d_gauss


def MovPoseDescriptor(p3d_gauss, StartFrame):

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

    ## Feature Vector
    f_v = np.concatenate((Pt0, vec, acc), axis=1)
    # print(f_v.shape[0])
    z = np.copy(f_v[f_v.shape[0] - 1, :])
    z = np.matlib.repmat(z, 4, 1)
    feat_vec = np.vstack((f_v, z))

    print(feat_vec.shape)
    return feat_vec


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

def DistMatPlot(sim_f_v, sim_path, name=None, flag=None, save_flag=None):

    if save_flag == 1:

        goal_dir = os.path.join(sim_path)
        fig, ax = plt.subplots()
        # cmap = cm.get_cmap('YlGnBu')
        cax = ax.matshow(sim_f_v, interpolation='None')
        ax.grid(True)
        plt.xlabel('frames')
        plt.ylabel('frames')

        if flag == 'similarity':
            my_file = name + '_sim_mat'
            plt.title('Self Similarity Matrix\n Moving Pose Descriptor')
        elif flag == 'compare':
            my_file = name + '_comp_mat'
            plt.title('Distance Matrix\n Comparison ' + name)

        fig.colorbar(cax)
        plt.close('all')
        # plt.show()
        print(goal_dir + my_file)
        fig.savefig(goal_dir + my_file)

    else:
        print('Passing Plot...')
        pass
