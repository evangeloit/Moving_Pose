from Moving_Pose_Descriptor import ComputeDatabase as cdb
from Moving_Pose_Descriptor import WeightedDistance as wd
from Moving_Pose_Descriptor import confmat as cfm
from Moving_Pose_Descriptor import Threshold_Precision_Recall as tpr
from Moving_Pose_Descriptor import frameDraw
import os
import numpy as np

#Paths
# dtpath = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/'
dtpath = '/home/evangeloit/Desktop/GitBlit_Master/Moving_Pose_Descriptor/Mydataset/data/'
# landmarks_path = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/rs/Human_tracking/results_camera_invariant/"
landmarks_path = "/home/evangeloit/Desktop/GitBlit_Master/Moving_Pose_Descriptor/Mydataset/res/results_camera_invariant/"
# model_name = 'mh_body_male_customquat'
model_name = 'mh_body_male_custom_1050'
dest_path = "/home/evangeloit/Desktop/GitBlit_Master/Moving_Pose_Descriptor/test_sequences/mydataset/"

# sflag =  0 : Turn off plots , 1: save figures to path. Global parameter
sflag = 0

# Similarity Matrix -- save Figure path
# #mhad
# savefig_sim = os.getcwd() + "/plots/mhad_dtset_plots_plots/similarity_matrix/"

#mydataset
savefig_sim = os.getcwd() + "/plots/mydataset_plots/similarity_matrix/"


# Compare one set with all the other datasets -- save Figure path
# #mhad
# savefig_comp = os.getcwd() + "/plots/mhad_dtset_plots/compare/"

# #My dataset
# savefig_comp = os.getcwd() + "/plots/mydataset_plots/compare/"

# DTW figures path
#mhad
# savefig_dtw = os.getcwd() + "/plots/mydataset_plots/dtw/"

#mydataset
savefig_dtw = os.getcwd() + "/plots/mydataset_plots/dtw/"
params_dtw = [0, savefig_dtw] # sflag 0 or 1 , savefig_dtw = path to save plot /

# Confusion matrix save figures path
# #mhad
# savefig_conf = os.getcwd() + "/plots/mhad_dtset_plots/confusion_matrix/"

# mydataset
savefig_conf = os.getcwd() + "/plots/mydataset_plots/confusion_matrix/"

#mhad
# savefig_evalmat = os.getcwd() + "/plots/mhad_dtset_plots/"

#mydataset
savefig_evalmat = os.getcwd() + "/plots/mydataset_plots/"
params_evalmat = [0, savefig_evalmat]

#Threshold precision recall savefig
#mhad
# savefig_tpr = os.getcwd() + "/plots/mhad_dtset_plots/TPR/"

#mydataset
savefig_tpr = os.getcwd() + "/plots/mydataset_plots/TPR/"
params_tpr = [0, savefig_tpr]

# subject_labels = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12"]

#Mydataset
subject_labels = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09"]
actions_labels = ["A01", "A02", "A03", "A04", "A05"]

# Choose most Confident frames for Classification
preprocess = False

################### CALLS #####################

# Full Database
database, fv_subj = cdb.db_construct(dtpath, landmarks_path, model_name, savefig=savefig_sim)

# # fullDatabase = cdb.db_lenOfseq(database)

nSubjects = 9
nActions = 4

if preprocess:
#####################################################################################################
    # Reduce database size by sub and act. Counting from 1
    reducedDatabase = cdb.db_reduce(database, nSubjects, nActions)

    # Add Length of Sequence to Database
    # database_lenofSeq = cdb.db_lenOfseq(database)
    database_lenofSeq = cdb.db_lenOfseq(reducedDatabase)

    # Compute confidence for every frame (KNN)/paramas:[train, test, relativeWindowsize(0 - 2), k nns]
    relativeWindow = 0.2
    k = 10
    # Assign confidence in every frame / BEST params for mhad : [ 0.93  0.9   0.1   0.45  6.  ]
    wvec = wd.wvector(1, 0.64, 0.3)
    # wvec = wd.wvector(0.93, 0.9, 0.45)

    conf_database = cdb.db_frameConfidence(database_lenofSeq, database_lenofSeq, relativeWindow, k, wvec)

    # Draw confidence on image
    # images = frameDraw.ConfidenceImage(dtpath, dest_path, conf_database, nSubjects, nActions)

    # Filter database by confidence[keep most confident frames] /export feature vector by sub for most conf frames
    keepConfidence = 0.8
    mostConf, fv_subj_modified = cdb.filter_byConfidence(conf_database, keepConfidence)
    # np.save('db_opencv_conf.npy',conf_database)
    np.save('fv_subj_conf.npy', fv_subj_modified)

    database_diff = conf_database.shape[0] - mostConf.shape[0]
    print "\nDatabase all frames: ", conf_database.shape[0]
    print "Database Conf frames: ", mostConf.shape[0]
    print "Frame Loss : ", database_diff
    print "Database Loss %.3f percent : " % ((float(database_diff)/conf_database.shape[0])*100)

else:

    # reduced feature vector by subject
    fv_subj_modified = fv_subj[0:nSubjects, 0:nActions]

##############################################################################################################

#Self Similarity Plot
# cdb.self_similarity(fv_subj_conf, actions_labels, subject_labels, savefig=savefig_sim)
# cdb.self_similarity(fv_subj_modified, actions_labels, subject_labels, savefig=savefig_sim)

#Compute DTW
evmat = cdb.computeDTW(fv_subj_modified, dtpath, actions_labels, sflag=sflag, params_dtw=params_dtw, savefig_conf=savefig_conf)

# Evaluation Matrix
confusion_matrix_all = cfm.evaluation_matrix(evmat, nSubjects, nActions, savefig_eval=params_evalmat)
np.save('eval_mat_new.npy', confusion_matrix_all)

#### Multiples Samples in training set #####

# Dataset Classification Score
class_score = cdb.classScore(confusion_matrix_all, nSubjects)

# Accuracy - Precision - Recall [Confusion_Matrix Total / Per Class]
mClassPerf = cdb.accuracy_multipleSample(confusion_matrix_all, nSubjects, nActions)

##### 1 Sample in Training Set #######

# Performance per class and Average. [iterations: Pick different random training samples in every iteration and compute aveage per class and average overall]
ClassPerformance, AveragePerformance = cdb.accuracy_oneSample(confusion_matrix_all, nSubjects, nActions, 1000)

# Threshold: 0:1:0.05 - Calculate Accuracy - Precision - Recall
# tpr.precision_recall(confusion_matrix_all, nSubjects, nActions, actions_labels, save_fig_tpr=params_tpr)

print()