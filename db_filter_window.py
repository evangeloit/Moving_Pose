import numpy as np

def db_window(windowSize, train, age):

    window_range = [age - windowSize/2, age + (windowSize+1)/2]

    keep_frames = []

    for iframe in range(0, train.shape[0]):

        if  window_range[0] <= train[iframe][3] & train[iframe][3] < window_range[1]:

            keep_frames.append(train[iframe])

    return np.array(keep_frames)


def db_window_relative(windowSizeRelative, train, age):

    keep_frames = []

    for iframe in range(0, train.shape[0]):

        windowSizeAbs = int(windowSizeRelative * train[iframe][4])

        window_range = [age - windowSizeAbs / 2, age + (windowSizeAbs + 1) / 2]

        if  window_range[0] <= train[iframe][3] & train[iframe][3] < window_range[1]:

            keep_frames.append(train[iframe])

    return np.array(keep_frames)