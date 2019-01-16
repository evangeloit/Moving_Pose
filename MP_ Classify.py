from Moving_Pose_Descriptor import ComputeDatabase as cdb
from Moving_Pose_Descriptor import WeightedDistance as wd
from Moving_Pose_Descriptor import confmat as cfm
from Moving_Pose_Descriptor import Threshold_Precision_Recall as tpr
import os
import numpy as np

#Paths
dtpath = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/'
landmarks_path = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/rs/Human_tracking/results_camera_invariant/"
model_name = 'mh_body_male_customquat'

# sflag =  0 : Turn off plots , 1: save figures to path. Global parameter
sflag = 0

# Similarity Matrix -- save Figure path
savefig_sim = os.getcwd() + "/plots/conf_matrix/MP_sim_mat/"

# Compare one set with all the other datasets -- save Figure path
savefig_comp = os.getcwd() + "/plots/conf_matrix/MP_comp_mat/"

# DTW figures path
savefig_dtw = os.getcwd() + "/plots/conf_matrix/dtw_res_conf/"
params_dtw = [0, savefig_dtw] # sflag 0 or 1 , savefig_dtw = path to save plot /

# Confusion matrix save figures path
savefig_conf = os.getcwd() + "/plots/conf_matrix/conf/"

# 1 vs all / Average dataset performance save figs path
savefig_avg = os.getcwd() + "/plots/conf_matrix/"
params_avg = [0, savefig_avg]

savefig_evalmat =  os.getcwd() + "/plots/conf_matrix/"
params_evalmat = [0, savefig_avg]

actions_labels = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11"]
# global actions_labels

################### CALLS #####################

#Full Database
database, fv_subj = cdb.db_construct(dtpath, landmarks_path, model_name)

# fullDatabase = cdb.db_lenOfseq(database)
#Reduce database size by sub and act. Counting from 1
subjects = 2
actions = 3
reducedDatabase = cdb.db_reduce(database, subjects, actions)

#Add Length of Sequence to Database
database_lenofSeq = cdb.db_lenOfseq(reducedDatabase)

#Compute confidence for every frame (KNN)/paramas:[train, test, relativeWindowsize(0 - 2), k nns]
relativeWindow = 1.4
k = 10
# Assign confidence in every frame / BEST params for mhad : [ 0.93  0.9   0.1   0.45  6.  ]
wvec = wd.wvector(1, 0.64, 0.3)
conf_database = cdb.db_frameConfidence(database_lenofSeq, database_lenofSeq, relativeWindow, k, wvec)

# np.save('db_opencv_conf.npy', conf_database)
#Plot Confidence to frame opencv



# # Filter database by confidence[keep most confident frames] /export feature vector by sub for most conf frames
# keepConfidence = 1.0
# mostConf, fv_subj_conf = cdb.filter_byConfidence(conf_database, keepConfidence)
#
# database_diff = conf_database.shape[0] - mostConf.shape[0]
# print("Database all frames: ", conf_database.shape[0])
# print("Database Conf frames: ", mostConf.shape[0])
# print("Frame Loss : ", database_diff)
#
# #Compute DTW
# evmat = cdb.computeDTW(fv_subj_conf, dtpath, sflag=sflag, params_dtw=params_dtw, savefig_conf=savefig_conf)
#
# #Evaluation Matrix
# # np.save('evmat.npy',evmat)
# eval_mat = cfm.evaluation_matrix(evmat, subjects, actions, savefig=params_evalmat)
# # np.save('eval_mat.npy',eval_mat)
#
# #Calculate Accuracy - Precision - Recall - Threshold: 0:1:0.05
# tpr.precision_recall(eval_mat, subjects, actions, actions_labels)


print()