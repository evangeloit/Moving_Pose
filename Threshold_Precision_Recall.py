import numpy as np
import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from Moving_Pose_Descriptor.heatmap import heatmap
from Moving_Pose_Descriptor.heatmap import annotate_heatmap
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
conf = np.load('eval_mat.npy')

conf = normalize(conf, axis=0, norm='max')

act_thres_pres_rec = np.zeros((11, 11, 2), dtype=float)

for t in range(0, 11):
    THRES = 0.1 * t

    gt_pred_conf = np.zeros((11, 11), dtype=int) # rows = predictec, cols = ground truth
    # THRES = 0.2

    # minimum of average errors
    ncorrect = 0
    for iRow in range(0, 132):
        #conf[iRow][iRow] = 1000000 # remove diagonal

        avg_err = np.zeros(11, dtype=float)

        # remove currect subject from comparison
        subjects = range(0, 12)
        subjects = np.setdiff1d(subjects, np.array([iRow % 12]))

        for iAction in range(0, 11):
            for iSubject in subjects:
                avg_err[iAction] += conf[iRow][iAction * 12 + iSubject] * (1.0/11)

        gt_answer = int(np.floor(iRow / 12))

        gt_pred_conf[gt_answer][np.argwhere(avg_err <= THRES)] += 1

    precision = np.zeros(11, dtype=float)
    recall    = np.zeros(11, dtype=float)
    rowise  = np.sum(gt_pred_conf, axis=0)
    colwise = np.sum(gt_pred_conf, axis=1)

    for iAction in range(0, 11):
        precision[iAction] = float(gt_pred_conf[iAction][iAction]) / (rowise[iAction] + 0.001)
        recall[iAction]    = float(gt_pred_conf[iAction][iAction]) / (colwise[iAction] + 0.001)

        act_thres_pres_rec[iAction][t][0] = precision[iAction]
        act_thres_pres_rec[iAction][t][1] = recall[iAction]

    # act_thres_prec_rec
#
# for iAction in range(0, 11):
#     # create new graph
#     for iThres in range(0, 11):
#         # add act_thres_pres_rec[iAction][iThres][0/1] to graph

print 'tat'
#
# thr = 0.2
# positives = 0
# negatives = 0
#
# for iRow in range (0,12):
#     RowConf = cmf_norm[iRow][:]
#     index = np.argwhere(RowConf<thr)
#
#     for idx in index:
#         # print(idx)
#         if idx<=11:
#             positives+=1
#         if idx>11:
#             negatives+=1
# # plt.show()
# print()