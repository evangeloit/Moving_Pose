import numpy as np
import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from Moving_Pose_Descriptor import FrameWiseClassify
from Moving_Pose_Descriptor import databaseModify
from Moving_Pose_Descriptor import db_filter_window
from Moving_Pose_Descriptor import WeightedDistance as wd
from Moving_Pose_Descriptor import confmat as cfm
from Moving_Pose_Descriptor import Threshold_Precision_Recall as tpr
import functools
import json

#
# #Paths
# dtpath = '/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/data/'
# landmarks_path = "/home/evangeloit/Desktop/GitBlit_Master/PythonModel3dTracker/Data/rs/Human_tracking/results_camera_invariant/"
# model_name = 'mh_body_male_customquat'
#
# # sflag =  0 : Turn off plots , 1: save figures to path. Global parameter
# sflag = 0
#
# # Similarity Matrix -- save Figure path
# savefig_sim = os.getcwd() + "/plots/conf_matrix/MP_sim_mat/"
#
# # Compare one set with all the other datasets -- save Figure path
# savefig_comp = os.getcwd() + "/plots/conf_matrix/MP_comp_mat/"
#
# # DTW figures path
# savefig_dtw = os.getcwd() + "/plots/conf_matrix/dtw_res_conf/"
# params_dtw = [0, savefig_dtw] # sflag 0 or 1 , savefig_dtw = path to save plot /
#
# # Confusion matrix save figures path
# savefig_conf = os.getcwd() + "/plots/conf_matrix/conf/"
#
# # 1 vs all / Average dataset performance save figs path
# savefig_avg = os.getcwd() + "/plots/conf_matrix/"
# params_avg = [0, savefig_avg]
#
# savefig_evalmat =  os.getcwd() + "/plots/conf_matrix/"
# params_evalmat = [0, savefig_avg]
#
# actions_labels = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11"]
# # global actions_labels

def db_construct(dtpath, landmarks_path, model_name):
    print("Constructing Database from files")
    """FEATURE VECTOR CALCULATION"""
    # Gaussian Filter Parameters
    sigma = 1
    w = 5  # windowSize
    t = (((w - 1) / 2) - 0.5) / sigma  # truncate

    # Feature Vector starting Frame
    StartFrame = 2  # Start from 3rd Frame's 3D coordinates

    # Subjects
    subj_name = mpt.AlpNumSorter(os.listdir(dtpath))  # List of Subjects in the directory
    # print(subj_name)

    #Num of actions in subject's path
    a, a_no_ext = mpt.list_ext(os.path.join(dtpath, subj_name[0]), 'json')
    acts = mpt.AlpNumSorter(a)
    num_of_acts = len(mpt.AlpNumSorter(a_no_ext))

    fv_all = []
    # Feature vector by subject initialiazation
    fv_subj = np.empty((len(subj_name), num_of_acts), np.dtype(np.object))

    for subj in range(0, len(subj_name)):# for every subject
        a, a_no_ext = mpt.list_ext(os.path.join(dtpath, subj_name[subj]), 'json')
        acts = mpt.AlpNumSorter(a)
        acts_no_ext = mpt.AlpNumSorter(a_no_ext)
        for act in range(0, len(acts)):  # for every action of a subject

            dataset_dir = os.path.join(dtpath, subj_name[subj], acts[act])
            input_dir = os.path.join(landmarks_path, acts_no_ext[act]+"_ldm.json")

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

            fv_all.append(feat_vec)

            # Build feature vector by subject
            fv_np = np.array(feat_vec)
            fv_subj[subj][act] = fv_np

    """DATABASE CONSTRUCTION"""
    #Construct and Save database
    database = []

    for iSubject in range(0, len(subj_name)):
        for iAction in range(0, num_of_acts):
            for iframe in range(0, len(fv_subj[iSubject][iAction])):
                dt = tuple((fv_subj[iSubject][iAction][iframe], iSubject, iAction, iframe))
                database.append(dt)

    database = np.array(database)



    np.save('newDatabase', database)

    return database ,fv_subj


def db_reduce(database, numofSubs, numofActs):
    print("Reducins Database in " + str(numofSubs) + " Subjects / "+ str(numofActs) + " Actions")
    numofActs = numofActs - 1
    numofSubs = numofSubs - 1
    # Reduce database to "Num" subjects "Num" Actions each
    keepframes = []
    for iframe in range(0, database.shape[0]):

        if database[iframe][1] <= numofSubs and database[iframe][2] <= numofActs:
            keepframes.append(iframe)

    data_reduced = database[keepframes]

    return data_reduced

def db_lenOfseq(database):
    print("Adding Length of Sequence for every frame in Database...")
    # Add Length of Sequence to Database
    lenOfsequence = [databaseModify.db_lengthOfSequence(database, database[iframe][1], database[iframe][2]) for iframe in range(0, database.shape[0])]
    lenOfsequence = np.array(lenOfsequence)

    database_lenofSeq = np.column_stack((database, lenOfsequence))

    return database_lenofSeq

def db_frameConfidence(database, db_test, relativeWindow, k, wvec):

    # # Assign confidence in every frame / BEST params for mhad : [ 0.93  0.9   0.1   0.45  6.  ]
    # wvec = wd.wvector(1, 0.64, 0.3)

    metric = functools.partial(wd.wdistance, wvec)
    # metric = distance.euclidean
    filter = functools.partial(db_filter_window.db_window_relative, relativeWindow)

    excl_flag = True
    confidence = []
    for iframe in range(0, db_test.shape[0]):

        if db_test[iframe][2] != db_test[iframe - 1][2]:
            excl_flag = True

        # exlude from database current subject's Action
        if excl_flag == True:
            print(db_test[iframe][1], db_test[iframe][2])
            db_exclude_act, excl_flag = databaseModify.db_exclude(database, db_test[iframe][1], db_test[iframe][2])

        # Create a new database restricted to a time window
        db = filter(db_exclude_act, db_test[iframe][3])

        frame = db_test[iframe]
        classconf = FrameWiseClassify.classifyKNN(frame[0], db, k, metric)

        confidence.append(classconf[1])

    confDatabase = np.column_stack((db_test, confidence))

    return confDatabase

def filter_byConfidence(conf_database, confidence):
    # filter frames by confidence
    keep_frames = []
    for iframe in range(0, conf_database.shape[0]):

        if conf_database[iframe][5] >= confidence:
            keep_frames.append(iframe)

    mostConf = conf_database[keep_frames]

    subjects = mostConf[-1, 1] + 1
    actions = mostConf[-1, 2] + 1

    # print(subjects)
    # print(actions)
    #Build Feature vector by subject with most confident frames
    fv_subj = np.zeros((subjects, actions), dtype=object)

    for iSubject in range(0, subjects):
        for iAction in range(0, actions):
            k = mostConf[(np.where((mostConf[:, 1] == iSubject) & (mostConf[:, 2] == iAction)))]
            k2 = []
            for inum in range(0, len(k)):
                k2.append(k[inum][0])
            k2 = np.array(k2)

            fv_subj[iSubject][iAction] = k2

    return mostConf, fv_subj

def computeDTW(fv_subj, dtpath, sflag=None,params_dtw=None ,savefig_conf=None):

    # print(fv_subj.shape[0], fv_subj.shape[1])

    subjects = fv_subj.shape[0]
    actions = fv_subj.shape[1]
    SubjectsActions = [subjects ,actions]

    # Subjects
    subj_name = mpt.AlpNumSorter(os.listdir(dtpath))  # List of Subjects in the directory
    subj_name = subj_name[0:subjects]
    print(subj_name)


    evmat = np.empty((subjects, subjects), np.dtype(np.object))


    for sub in range(0, subjects):
        ct = 0
        for sub2 in range(0, subjects):

            # Subjects from subj_name list
            subject1 = subj_name[sub]
            subject2 = subj_name[ct]

            # Feature Vectors by subject
            fv_1 = fv_subj[sub]
            fv_2 = fv_subj[ct]

            ct = ct + 1
            # Create confusion matrix for every pair of subjects

            score, class_score, missclass = cfm.Conf2Subject(subject1, subject2, SubjectsActions, dtpath, fv_1, fv_2,params=params_dtw)
            evmat[sub][sub2] = score

            if sflag == 1:
                params_cmf = [score, actions, class_score, missclass, sflag, savefig_conf]
                cfm.cfm_savefig(subject1, subject2, params_cmf)
    return evmat

# ################### CALLS #####################
#
# #Full Database
# database, fv_subj = db_construct(dtpath, landmarks_path, model_name)
#
# #Reduce database size by sub and act. Counting from 1
# subjects = 5
# actions = 5
# reducedDatabase = db_reduce(database, subjects, actions)
#
# #Add Length of Sequence to Database
# database_lenofSeq = db_lenOfseq(reducedDatabase)
#
# #Compute confidence for every frame (KNN)/paramas:[train, test, relativeWindowsize(0 - 2), k nns]
# relativeWindow = 1.4
# k = 6
# # Assign confidence in every frame / BEST params for mhad : [ 0.93  0.9   0.1   0.45  6.  ]
# wvec = wd.wvector(1, 0.64, 0.3)
# conf_database = db_frameConfidence(database_lenofSeq, database_lenofSeq, relativeWindow, k, wvec)
#
# # Filter database by confidence[keep most confident frames] /export feature vector by sub for most conf frames
# keepConfidence = 1.0
# mostConf, fv_subj_conf = filter_byConfidence(conf_database, keepConfidence)
#
# database_diff = conf_database.shape[0] - mostConf.shape[0]
# print("Database all frames: ", conf_database.shape[0])
# print("Database Conf frames: ", mostConf.shape[0])
# print("Frame Loss : ", database_diff)
#
# #Compute DTW
# evmat = computeDTW(fv_subj_conf, dtpath, sflag=sflag, params_dtw=params_dtw, savefig_conf=savefig_conf)
#
# #Evaluation Matrix
# # np.save('evmat.npy',evmat)
# eval_mat = cfm.evaluation_matrix(evmat, subjects, actions, savefig=params_evalmat)
# # np.save('eval_mat.npy',eval_mat)
#
# #Calculate Accuracy - Precision - Recall - Threshold: 0:1:0.05
# tpr.precision_recall(eval_mat, subjects, actions, actions_labels)
#
#
# print()