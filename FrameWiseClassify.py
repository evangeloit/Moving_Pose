# import numpy as np
# from scipy.spatial import distance as dst
from Moving_Pose_Descriptor import MP_tools2 as mpt

# @frame is one feature vector [Px, Py, Pz, mVel , mAcc] x 14 joints
# @database [N][[feature vector...], subject, action]
def classifyKNN(frame, database, k, metric):
    distances = []

    for iframe in range(0, database.shape[0]):
        d = metric(frame, database[iframe][0])
        distances.append((d, database[iframe][1], database[iframe][2]))

    distances.sort(key=lambda row: row[0])

    return mpt.most_often_occurence(distances[0 : k])