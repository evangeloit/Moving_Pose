import numpy as np
import math
from Moving_Pose_Descriptor import ComputeDatabase as cdb
from Moving_Pose_Descriptor import confmat as cfm
import os
from Moving_Pose_Descriptor import Threshold_Precision_Recall as tpr

confusion_matrix = np.load('eval_mat_new.npy')
subjects = 12
actions = 11

# range_max = 0
# range_min = 0

np.fill_diagonal(confusion_matrix, float('inf'))
hits = 0
for row in enumerate(confusion_matrix):
    answer = np.argmin(row[1], axis=0) / subjects
    correct = row[0] / subjects

    if answer == correct:
        hits += 1

    # iCorrectAction = row[0] / subjects
    # correctMin = iCorrectAction * subjects
    # correctMax = correctMin + subjects
    #
    # if answer >= correctMin and answer < correctMax:
    #     hits += 1


    # action_change = row[0] % subjects
    #
    # if action_change == 0:
    #     range_min = row[0]
    #     range_max = row[0] + subjects
    #
    # mincol_idx = np.argmin(row[1], axis=0)# min column index
    #
    # if mincol_idx < range_max and mincol_idx >= range_min:
    #     hits += 1


class_score = (float(hits) / confusion_matrix.shape[0]) * 100
print "Class Score : %f" %(class_score)
