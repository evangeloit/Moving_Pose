import numpy as np
from Moving_Pose_Descriptor import FrameWiseClassify as FrameWiseClassify
from Moving_Pose_Descriptor import db_filter_window
from Moving_Pose_Descriptor import WeightedDistance as wd
import functools
from scipy.spatial import distance

# Load Complete database
database = np.load("db_frame_subject_action_age.npy")

# Reduce database to 4 subjects 4 Actions each
keepframes = []
for iframe in range(0, database.shape[0]):

    if database[iframe][1] <= 4 and database[iframe][2] <= 4:
        keepframes.append(iframe)

data4subs = database[keepframes]
test4subs = data4subs.copy()

# Assign confidence in every frame / BEST params for mhad : [ 0.93  0.9   0.1   0.45  6.  ]
wvec = wd.wvector(1, 0.64, 0.3)

metric = functools.partial(wd.wdistance, wvec)
# metric = distance.euclidean
filter = functools.partial(db_filter_window.db_window, 100)

excl_flag = True
results = []
for iframe in range(0, test4subs.shape[0]):

    if test4subs[iframe][2] != test4subs[iframe-1][2]:
        excl_flag = True

    # exlude from database current subject's Action
    if excl_flag == True:
        print(test4subs[iframe][1], test4subs[iframe][2])
        db_exclude_act, excl_flag = db_filter_window.db_exclude(data4subs, test4subs[iframe][1], test4subs[iframe][2])

    # Create a new database restricted to a time window
    db = filter(db_exclude_act, test4subs[iframe][3])

    frame = data4subs[iframe]
    classconf = FrameWiseClassify.classifyKNN(frame[0], db, 3, metric)

    res = [classconf[1]]
    results.append(res)

results = np.array(results)
confDatabase = np.append(data4subs, results, axis=1)

# confDataSubject

np.save('data4subs_confidence.npy',confDatabase)
print()