from Moving_Pose_Descriptor import ComputeDatabase as cdb
from Moving_Pose_Descriptor import WeightedDistance as wd
from Moving_Pose_Descriptor import confmat as cfm
from Moving_Pose_Descriptor import Threshold_Precision_Recall as tpr
from Moving_Pose_Descriptor import frameDraw
import os
import numpy as np

#Paths
dtpath = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/'
landmarks_path = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/rs/Human_tracking/results_camera_invariant/"
model_name = 'mh_body_male_customquat'
dest_path = "/home/evangeloit/Desktop/GitBlit_Master/Moving_Pose_Descriptor/test_sequences/"

# sflag =  0 : Turn off plots , 1: save figures to path. Global parameter
sflag = 1

# Similarity Matrix -- save Figure path
# #mhad
# savefig_sim = os.getcwd() + "/plots/conf_matrix/MP_sim_mat/"

#mydataset
savefig_sim = os.getcwd() + "/plots/mydataset_plots/similarity_matrix/"


# Compare one set with all the other datasets -- save Figure path
# #mhad
# savefig_comp = os.getcwd() + "/plots/conf_matrix/MP_comp_mat/"

#My dataset
savefig_comp = os.getcwd() + "/plots/mydataset_plots/compare/"

# DTW figures path
# #mhad
# savefig_dtw = os.getcwd() + "/plots/conf_matrix/dtw_res_conf/"
# params_dtw = [0, savefig_dtw] # sflag 0 or 1 , savefig_dtw = path to save plot /

#mydataset
savefig_dtw = os.getcwd() + "/plots/mydataset_plots/dtw/"
params_dtw = [0, savefig_dtw] # sflag 0 or 1 , savefig_dtw = path to save plot /

# Confusion matrix save figures path
# #mhad
# savefig_conf = os.getcwd() + "/plots/conf_matrix/conf/"

# mydataset
savefig_conf = os.getcwd() + "/plots/mydataset_plots/confusion_matrix/"


#All vs All matrix
# #mhad
# savefig_evalmat =  os.getcwd() + "/plots/conf_matrix/"
# params_evalmat = [0, savefig_avg]

#mydataset
savefig_evalmat = os.getcwd() + "/plots/mydataset_plots/"
params_evalmat = [1, savefig_evalmat]


#Threshold precision recall savefig
savefig_tpr = os.getcwd() + "/plots/mydataset_plots/TPR/"
params_tpr = [0, savefig_tpr]

subject_labels = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12"]
actions_labels = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11"]


################### CALLS #####################

# Full Database
database, fv_subj = cdb.db_construct(dtpath, landmarks_path, model_name, savefig=savefig_sim)

# # fullDatabase = cdb.db_lenOfseq(database)
# # Reduce database size by sub and act. Counting from 1
subjects = 12
actions = 11
reducedDatabase = cdb.db_reduce(database, subjects, actions)
#
# #reduced feature vector by subject
# reduced_fv_subj = fv_subj[0:subjects, 0:actions]

#####################################################################################################
# Add Length of Sequence to Database
# database_lenofSeq = cdb.db_lenOfseq(database)
database_lenofSeq = cdb.db_lenOfseq(reducedDatabase)


# Compute confidence for every frame (KNN)/paramas:[train, test, relativeWindowsize(0 - 2), k nns]
relativeWindow = 1.4
k = 10
# Assign confidence in every frame / BEST params for mhad : [ 0.93  0.9   0.1   0.45  6.  ]
wvec = wd.wvector(1, 0.64, 0.3)
conf_database = cdb.db_frameConfidence(database_lenofSeq, database_lenofSeq, relativeWindow, k, wvec)

# np.save('db_opencv_conf.npy',conf_database)

# Draw confidence on image
# images = frameDraw.ConfidenceImage(dtpath, dest_path, conf_database, subjects, actions)

# Filter database by confidence[keep most confident frames] /export feature vector by sub for most conf frames
keepConfidence = 0.8
mostConf, fv_subj_conf = cdb.filter_byConfidence(conf_database, keepConfidence)
# np.save('db_opencv_conf.npy',conf_database)
np.save('fv_subj_conf.npy', fv_subj_conf)

database_diff = conf_database.shape[0] - mostConf.shape[0]
print("Database all frames: ", conf_database.shape[0])
print("Database Conf frames: ", mostConf.shape[0])
print("Frame Loss : ", database_diff)
print("Database Loss % : ", (float(database_diff)/conf_database.shape[0])*100)

##############################################################################################################

#Self Similarity Plot

# cdb.self_similarity(fv_subj_conf, actions_labels, subject_labels, savefig=savefig_sim)
# cdb.self_similarity(reduced_fv_subj, actions_labels, subject_labels, savefig=savefig_sim)

#Compute DTW
evmat = cdb.computeDTW(fv_subj_conf, dtpath, actions_labels, sflag=sflag, params_dtw=params_dtw, savefig_conf=savefig_conf)
# evmat = cdb.computeDTW(reduced_fv_subj, dtpath, actions_labels, sflag=sflag, params_dtw=params_dtw, savefig_conf=savefig_conf)

#Evaluation Matrix
# np.save('evmat.npy',evmat)
eval_mat = cfm.evaluation_matrix(evmat, subjects, actions, savefig_eval=params_evalmat)
np.save('eval_mat_new.npy', eval_mat)

#Calculate Accuracy - Precision - Recall - Threshold: 0:1:0.05
tpr.precision_recall(eval_mat, subjects, actions, actions_labels, save_fig_tpr=params_tpr)


print()