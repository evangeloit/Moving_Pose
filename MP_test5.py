import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import os
import MP_tools as mpt

dataset = ['mhad_s01_a04', 'mhad_s02_a04', 'mhad_s03_a04','mhad_s04_a04'\
          ,'mhad_s05_a04','mhad_s06_a04','mhad_s07_a04','mhad_s08_a04',\
           'mhad_s09_a01','mhad_s10_a04', 'mhad_s11_a04','mhad_s12_a04',]
model_name = 'mh_body_male_customquat'

# Gaussian Filter Parameters
sigma = 1
w = 5  # windowSize
t = (((w - 1) / 2) - 0.5) / sigma  # truncate

# Similarity Matrix save Figure path
savefig_sim = os.getcwd() + "/plots/MP_Similarity_Matrix/"
# Compare one set with all the other datasets -- save Figure path
savefig_comp = os.getcwd() + "/plots/MP_Similarity_Matrix/comparisons/"

FV_new = []

for name in dataset:

    ## Load data from Json ##
    dataPoints, dataLim = mpt.load_data(name)

    init_frame = dataLim['limits'][0]
    last_frame = dataLim['limits'][1]

    ##### Create 3D points array #####

    p3d = mpt.Create3dPoints(init_frame, last_frame, dataPoints, model_name)

    ## Gaussian Filter 5 by 1 in time dimension


    p3d_gauss = mpt.GaussFilter3dPoints(p3d, sigma, t)

    #### Derivatives Calculation ####

    StartFrame = 2  # Start from 3rd Frame's 3D points# pt1 = np.array([])

    feat_vec = mpt.MovPoseDescriptor(p3d_gauss, StartFrame)

    FV_new.append(feat_vec)

    ## Similarity Matrix ##

    sim_f_v = squareform(pdist(feat_vec))

    ## Similarity - Plot ##

    mpt.SimilarityPlot(sim_f_v, savefig_sim, name=name, flag='similarity')


# Feature Vector Array for all datasets
fv_new = np.array(FV_new)

## Comparison of s01a04 Sim_Matrix with the all the other subjexts matrices ####

for subject in range(0, len(dataset)):
    Y = cdist(fv_new[2], fv_new[subject], 'euclidean')
    mpt.SimilarityPlot(Y, savefig_comp, name=dataset[subject], flag='compare')

# # Gaussian Smoothing - Plot ##
# mpt.smoothPlot(p3d, p3d_gauss)
