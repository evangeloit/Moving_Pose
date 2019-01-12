import numpy as np
from Moving_Pose_Descriptor import FrameWiseClassify
from Moving_Pose_Descriptor import databaseModify
from Moving_Pose_Descriptor import db_filter_window
from Moving_Pose_Descriptor import WeightedDistance as wd
import functools
# from scipy.spatial import distance

# # Load Complete database
# database = np.load("db_frame_subject_action_age.npy")
#
# # Reduce database to 4 subjects 4 Actions each
# keepframes = []
# for iframe in range(0, database.shape[0]):
#
#     if database[iframe][1] <= 4 and database[iframe][2] <= 4:
#         keepframes.append(iframe)
#
# # data4subs = database[keepframes]
# # test4subs = data4subs.copy()

data4subs = np.load('data4subs_lenOfSeq.npy')
test4subs = data4subs.copy()

# Assign confidence in every frame / BEST params for mhad : [ 0.93  0.9   0.1   0.45  6.  ]
wvec = wd.wvector(1, 0.64, 0.3)

metric = functools.partial(wd.wdistance, wvec)
# metric = distance.euclidean
filter = functools.partial(db_filter_window.db_window_relative, 0.7)

excl_flag = True
confidence = []
for iframe in range(0, test4subs.shape[0]):

    if test4subs[iframe][2] != test4subs[iframe-1][2]:
        excl_flag = True

    # exlude from database current subject's Action
    if excl_flag == True:
        print(test4subs[iframe][1], test4subs[iframe][2])
        db_exclude_act, excl_flag = databaseModify.db_exclude(data4subs, test4subs[iframe][1], test4subs[iframe][2])

    # Create a new database restricted to a time window
    db = filter(db_exclude_act, test4subs[iframe][3])

    frame = data4subs[iframe]
    classconf = FrameWiseClassify.classifyKNN(frame[0], db, 3, metric)

    confidence.append(classconf[1])


# #Add 5th column to database with the total Length of the sequence
# lenOfsequence = [databaseModify.db_lengthOfSequence(data4subs, test4subs[iframe][1], test4subs[iframe][2]) for iframe in range(0, test4subs.shape[0])]
#
#
# lenOfsequence = np.array(lenOfsequence)
# confidence = np.array(confidence)
confDatabase = np.column_stack((data4subs, confidence))

#Save Database
np.save('data4subs_lenOfSeq_confidence.npy', confDatabase)
print()