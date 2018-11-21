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
import Moving_Pose_Descriptor.MP_tools2 as mpt
import dpcore #dtw c
import matplotlib.pyplot as plt
from Moving_Pose_Descriptor.heatmap import heatmap
from Moving_Pose_Descriptor.heatmap import annotate_heatmap

# Controllers

# dataset = ['mhad_s01_a04', 'mhad_s02_a04', 'mhad_s03_a04','mhad_s04_a04'\
#             ,'mhad_s05_a04', 'mhad_s06_a04', 'mhad_s07_a04','mhad_s08_a04','mhad_s09_a01','mhad_s10_a04', 'mhad_s11_a04', 'mhad_s12_a04']
dataset_s1 = ['mhad_s01_a01','mhad_s01_a02', 'mhad_s01_a03','mhad_s01_a04',\
          'mhad_s01_a05', 'mhad_s01_a06','mhad_s01_a07','mhad_s01_a08',\
          'mhad_s01_a09','mhad_s01_a10','mhad_s01_a11','mhad_s06_a01',\
            'mhad_s06_a02', 'mhad_s06_a03','mhad_s06_a04', 'mhad_s06_a05',\
          'mhad_s06_a06','mhad_s06_a07','mhad_s06_a08','mhad_s06_a09','mhad_s06_a10','mhad_s06_a11']

# dataset_s6 = ['mhad_s06_a01','mhad_s06_a02', 'mhad_s06_a03','mhad_s06_a04', 'mhad_s06_a05',\
#           'mhad_s06_a06','mhad_s06_a07','mhad_s06_a08','mhad_s06_a09', 'mhad_s06_a10', 'mhad_s06_a11']

model_name = 'mh_body_male_customquat'

# Gaussian Filter Parameters
sigma = 1
w = 5  # windowSize
t = (((w - 1) / 2) - 0.5) / sigma  # truncate

# Feature Vector starting Frame
StartFrame = 2  # Start from 3rd Frame's 3D coordinates

# Similarity Matrix -- save Figure path
savefig_sim = os.getcwd() + "/plots/conf_matrix/MP_sim_mat/"

# Compare one set with all the other datasets -- save Figure path
savefig_comp = os.getcwd() + "/plots/conf_matrix/MP_comp_mat/"

# DTW figures path
savefig_dtw = os.getcwd() + "/plots/conf_matrix/dtw_res_conf/"

# sflag =  0 : Turn off plots , 1: save figures to path
sflag = 0

FV_new = []

for name in dataset_s1:

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

    feat_vec, vec, acc = mpt.MovPoseDescriptor(p3d_gauss, StartFrame)

    FV_new.append(feat_vec)


# Feature Vector Array for all datasets
fv_new = np.array(FV_new).copy()

## Similarity Matrix ##
for fv in range(0, len(fv_new)):
    sim_f_v = squareform(pdist(fv_new[fv]))

    ## Similarity - Plot ##
    mpt.DistMatPlot(sim_f_v, savefig_sim, name=dataset_s1[fv], flag='similarity', save_flag=sflag)


score = np.empty((len(dataset_s1)/2, len(dataset_s1)/2), np.dtype(np.float32))

## Comparison of s01a03 Feat Vector with the all the other datasets Feat_Vecs ####

for subject1 in range(0, len(dataset_s1)/2):
    for subject6 in range(0, len(dataset_s1)/2):

        Y = cdist(fv_new[subject1], fv_new[subject6+(len(dataset_s1)/2)], 'euclidean')
        p, q, C, phi = mpt.dtwC(Y, 0.1)

        # Sum of diagonal steps
        dsteps = 0
        for r in range(0, len(q)-1):
            qdot = abs(q[r] - q[r+1])
            pdot = abs(p[r] - p[r+1])
            s = qdot + pdot
            if s==2:
                dsteps = dsteps +1
                # print(dsteps)

        mpt.DistMatPlot(Y, savefig_comp, name=dataset_s1[subject1]+"||"+dataset_s1[subject6+(len(dataset_s1)/2)], flag='compare', save_flag=sflag)
        mpt.DistMatPlot(Y, savefig_dtw, q, p, name=dataset_s1[subject1]+"||"+dataset_s1[subject6+(len(dataset_s1)/2)], flag='DTW', save_flag=sflag)

        #Scores of DTW for every subject
        score[subject1][subject6] = (C[-1, -1]/((Y.shape[0]+Y.shape[1]))+dsteps)
        # score[subject1][subject6] = C[-1, -1]/dsteps


#mis/classification Score %
Pscore = ((score/np.amax(score))*100).copy()
Pmin = np.argmin(Pscore, axis=1)
Pvec = np.arange(0,len(Pmin))
m = Pmin == Pvec
mnot= Pmin != Pvec

missclass = np.sum(mnot.astype(int))
class_score = (np.sum(m.astype(int),dtype=float)/len(Pscore))*100

# Confusion Matrix
# actionsS1 = ["A02","A03","A04","A05","A06","A09"]
# actionsS6 = ["A02","A03","A04","A05","A06","A09"]

actionsS1 = ["A01","A02","A03","A04","A05","A06","A07","A08","A09","A10","A11"]
actionsS6 = ["A01","A02","A03","A04","A05","A06","A07","A08","A09","A10","A11"]

fig, ax = plt.subplots()
ax.set_xlabel('S06')
ax.set_ylabel('S01')
# ax.xaxis.set_label_position('top')

im, cbar = heatmap(Pscore, actionsS1, actionsS6,ax=ax,
                   cmap=None, cbarlabel="Score")

texts = annotate_heatmap(im, valfmt="{x:.2f} ")
plt.title('Classification Score = '+ str(class_score.round(decimals=2))+'%\nmisclassified='+str(missclass))
plt.rcParams.update({'font.size': 13})
fig.tight_layout()
plt.show()

print()

