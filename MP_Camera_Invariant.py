# MP_test5 :
# 1) Imports (3d) landmarks results from json
# 2) Creates an array of the (3D) landmarks
# 3) Filters all landmarks with a gaussian filter 5by1 in time dimension
# 4) Calculates Feature Vector for every given dataset
# 5) Creates a self Similarity matrix and saves it in a figure on demand (sflag)
# 6) Compares the feature vector of a given set with a different set using cdist
# and saves distance matrix figure on demand (sflag)

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import os
import MP_tools2 as mpt
import dpcore #dtw c
import matplotlib.pyplot as plt

# Controllers

dataset = ['mhad_s01_a04', 'mhad_s02_a04', 'mhad_s03_a04','mhad_s04_a04'\
            ,'mhad_s05_a04', 'mhad_s06_a04', 'mhad_s07_a04', 'mhad_s08_a04',\
           'mhad_s09_a01', 'mhad_s10_a04', 'mhad_s11_a04', 'mhad_s12_a04']

model_name = 'mh_body_male_customquat'

# Gaussian Filter Parameters
sigma = 1
w = 5  # windowSize
t = (((w - 1) / 2) - 0.5) / sigma  # truncate

# Feature Vector starting Frame
StartFrame = 2  # Start from 3rd Frame's 3D coordinates

# Similarity Matrix -- save Figure path
savefig_sim = os.getcwd() + "/plots/MP_Similarity_Matrix/"

# Compare one set with all the other datasets -- save Figure path
savefig_comp = os.getcwd() + "/plots/MP_Similarity_Matrix/comparisons/"

# DTW figures path
savefig_dtw = os.getcwd() + "/plots/MP_Similarity_Matrix/dtw_results/"

# sflag =  0 : Turn off plots , 1: save figures to path
sflag = 1

FV_new = []

for name in dataset:

    dataset_dir = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/" + name + ".json"
    input_dir = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/rs/Human_tracking/" \
                + name + "_results_ldm.json"

    ## Load data from Json ##
    dataPoints, dataLim = mpt.load_data(input_dir, dataset_dir)

    init_frame = dataLim['limits'][0]
    last_frame = dataLim['limits'][1]

    ##### Create 3D points array #####

    p3d = mpt.Create3dPoints(init_frame, last_frame, dataPoints, model_name)

    ## Gaussian Filter 5 by 1 in time dimension

    p3d_gauss = mpt.GaussFilter3dPoints(p3d, sigma, t)

    #### Create Feature Vector ####

    feat_vec = mpt.MovPoseDescriptor(p3d_gauss, StartFrame)

    FV_new.append(feat_vec)

    ## Similarity Matrix ##

    sim_f_v = squareform(pdist(feat_vec))

    ## Similarity - Plot ##

    mpt.DistMatPlot(sim_f_v, savefig_sim, name=name, flag='similarity', save_flag=sflag)


# Feature Vector Array for all datasets
fv_new = np.array(FV_new)
# print(fv_new.shape)
# Cparam = []
## Comparison of s01a03 Feat Vector with the all the other datasets Feat_Vecs ####
for subject in range(0, len(dataset)):
    Y = cdist(fv_new[4], fv_new[subject], 'euclidean')
    p, q, C, phi = mpt.dtwC(Y, 0.1)
    # c = C/(fv_new[subject].shape[0]*fv_new[subject].shape[1])
    # Cparam.append(c)
    mpt.DistMatPlot(Y, savefig_comp, name=dataset[subject], flag='compare', save_flag=sflag)
    mpt.DistMatPlot(Y, savefig_dtw, q, p, name=dataset[subject], flag='DTW', save_flag=sflag)

# print()