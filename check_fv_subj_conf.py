import numpy as np
from Moving_Pose_Descriptor import ComputeDatabase as cdb
from Moving_Pose_Descriptor import confmat as cfm
import os
from Moving_Pose_Descriptor import Threshold_Precision_Recall as tpr
# import matplotlib.pyplot as plt


dtpath = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/'

# sflag =  0 : Turn off plots , 1: save figures to path. Global parameter
sflag = 0

#mydataset
savefig_evalmat = os.getcwd() + "/plots/mydataset_plots/"
params_evalmat = [1, savefig_evalmat]

#mydataset
savefig_dtw = os.getcwd() + "/plots/mydataset_plots/dtw/"
params_dtw = [0, savefig_dtw] # sflag 0 or 1 , savefig_dtw = path to save plot /

#mydataset
savefig_conf = os.getcwd() + "/plots/mydataset_plots/confusion_matrix/"

#Threshold precision recall savefig
savefig_tpr = os.getcwd() + "/plots/mydataset_plots/TPR/"
params_tpr = [1, savefig_tpr]

subject_labels = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12"]
actions_labels = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11"]


fv_subj_conf = np.load('fv_subj_conf.npy')


subjects = fv_subj_conf.shape[0]
actions = fv_subj_conf.shape[1]


#Compute DTW
evmat = cdb.computeDTW(fv_subj_conf, dtpath, actions_labels, sflag=sflag, params_dtw=params_dtw, savefig_conf=savefig_conf)


#Evaluation Matrix
eval_mat = cfm.evaluation_matrix(evmat, subjects, actions, savefig_eval=params_evalmat)
np.save('eval_mat_new.npy', eval_mat)

tpr.precision_recall(eval_mat, subjects, actions, actions_labels, save_fig_tpr=params_tpr)
print()