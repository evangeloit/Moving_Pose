import numpy as np
from scipy.spatial import distance as dst
from Moving_Pose_Descriptor import FrameWiseClassify as FrameWiseClassify
from Moving_Pose_Descriptor import databaseModify
from Moving_Pose_Descriptor import WeightedDistance as wd
from Moving_Pose_Descriptor import db_filter_window
from Moving_Pose_Descriptor import noise
import functools
import time

# @filter_func function used to filter train, expecting two arguments (train, age, action)
def benchmark(train, filter_func, test, k, metric, age_uncertainty_in_frames=0):
    nCorrect = 0.0
    score = 0.0

    for iFrame in range(0, test.shape[0]):
        # TODO: Add noise

        db = filter_func(train, noise.addnoise(test[iFrame][3],age_uncertainty_in_frames))

        frame = test[iFrame]
        classconf = FrameWiseClassify.classifyKNN(frame[0], db, k, metric)

        if classconf[0] == frame[2]:
            nCorrect += 1
            score += classconf[1]
        else:
            score -= classconf[1]

    return (
        nCorrect / test.shape[0],
        score / test.shape[0]
    )


database = np.load("db_frame_subject_action_age.npy")

train, test = databaseModify.reduceDatabase(database, 10000, 400)
#
wvec = wd.wvector(0.4, 0.64, 0.3)

metric = functools.partial(wd.wdistance, wvec)

filter = functools.partial(db_filter_window.db_window, 200)

# start_time = time.time()
result = benchmark(train, filter, test, 3, metric, 15)

print(result)
exit()

# BEST params for mhad with windowSize:200 and noise level :15
# [Accuracy0.91, wPos : 0.4 ,wVel : 0.64, wAcc : 0.3, k : 3]

results = []
# iRow = 0
# age_uncertainty_in_frames = [5, 10, 15]
#
# for uncertainty in [0, 5, 10, 15]:
#
#     for wsize in range(100, 1000, 5):


for wpos in range(1, 11):
        for wvel in range(1, 11):
            print("%d percent done" % (10 * wpos + wvel - 11))

            for wacc in range(1, 11):
                for k in range(3, 11, 1):
                    wp = wpos * 0.1
                    wv = wvel * 0.1
                    wa = wacc * 0.1

                    wvec = wd.wvector(wp, wv, wa)

                    metric = functools.partial(wd.wdistance, wvec)

                    filter = functools.partial(db_filter_window.db_window, 200)

                    # start_time = time.time()
                    result = benchmark(train, filter, test, k, metric, 15)

                    # print("--- %s seconds ---" % (time.time() - start_time))
                    # exit()

                    res = [result[0], wp, wv, wa, k]
                    results.append(res)


results = np.array(results)
np.save('best_params_for_High_noise_test2.npy', results)
print()