import random
import numpy as np
import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
import json

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

def db_from_path(dtpath, export=None):
    subj_name = mpt.AlpNumSorter(os.listdir(dtpath))  # List of Subjects in the directory
    mydataset = []

    for subj in enumerate(subj_name):  # for every subject
        a, a_no_ext = mpt.list_ext(os.path.join(dtpath, subj[1]), 'json')
        acts = mpt.AlpNumSorter(a)
        acts_no_ext = mpt.AlpNumSorter(a_no_ext)
        for act in enumerate(acts_no_ext):
            mydataset.append(act[1])

    mydata = mpt.AlpNumSorter(mydataset)

    if export:
        with open('database_out.json', 'w') as outfile:
            json.dump(mydata, outfile)

    print(mydata)

    return mydata