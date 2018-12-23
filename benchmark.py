import numpy as np
from scipy.spatial import distance as dst
from Moving_Pose_Descriptor import FrameWiseClassify as FrameWiseClassify
from Moving_Pose_Descriptor import databaseClass
import functools

def sum(a, b):
    return a + b

class Sum:
    # members:
    # x

    def __init__(self, x):
        self.x = x

    def __call__(self, param):
        return sum(self.x, param)

s = Sum(23)

print(s(15))

s2 = functools.partial(sum, 23)

print(s2(15))

exit()

# @filter_func function used to filter train, expecting two arguments (train, age, action)
def benchmark(train, filter_func, test, k, metric):
    nCorrect = 0.0
    score = 0.0

    for iFrame in range(0, test.shape[0]):
        db = filter_func(train, iFrame, test[iFrame][2])
        # TODO: Once ids have been added to db change above line into:
        # db = filter_func(train, test[iFrame][3], test[iFrame][2])

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

# TODO:
# - Document feature vector format
# - Implement different distance metrics
#   - sum of absolute distances
#   - sum of squares
#   - certain aspects only (ie velocity only)
#   - weighted aspects
# - evaluate
# for w1 in range(0, 11):
#     for w2 in range(0, 11):
#         for w3 in range(0, 11):
#             w1n = w1 * 0.1
#             w2n = w2 * 0.1
#             w3n = w3 * 0.1
#
#             func = functools.partial(dist_vel_only, w1n, w2n, w3n)
#
#             benchmark(train, test, k, func)

# TODO: Add id(aka age) to frames in DB
database = np.load("db_frame_subject_action.npy")

# TODO: Bump up train and test counts to (10k, 200)
train, test = databaseClass.reduceDatabase(database, 1000, 100)
# TODO: Find optimal k!
k = 20

# @age age of incoming frame=(fv, sub, act, id)
# @action action of incoming frame
#def filter(windowSize, train, age, action):
    # compute (iFrame-ws, iFrame+ws)
    # output = empty()
    # for each frame in train with frame.action == action && frame.id \in window
    #   select frames that lie inisde window, add to output
    # return output

# for windowSizes in range(...):
#     f = functools.partial(filter, windowSize)
#    benchmark(train, f, ...)

for k in range(5, 21, 5):
    result = benchmark(train, lambda x, y, z: x, test, k, dst.euclidean)

    print("K: %s" % k)
    print("Accuracy: %s" % result[0])
    print("Confidence: %s" % result[1])