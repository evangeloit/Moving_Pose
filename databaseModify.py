import random
import numpy as np


# @fullDatabase = array[N][[features...], subject, action]
def reduceDatabase(fullDatabase, trainCount, testCount):
    indices = range(0, len(fullDatabase))
    random.shuffle(indices)

    return (
        fullDatabase[indices[0 : trainCount]],
        fullDatabase[indices[trainCount : trainCount + testCount]]
    )


def db_exclude(database, iSubject, iAction):

    keep_frames = []

    for iframe in range(0, database.shape[0]):

        if database[iframe][2] != iAction or database[iframe][1] != iSubject:
            keep_frames.append(iframe)

    new = database[keep_frames]

    excl_flag = False

    return new, excl_flag


def db_lengthOfSequence(database, iSubject, iAction):

    keep_frames = [iframe for iframe in range(0, database.shape[0]) if database[iframe][2] == iAction and database[iframe][1] == iSubject]

    length_of_sequence = len(keep_frames)

    return length_of_sequence
