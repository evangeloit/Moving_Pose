import numpy as np
from scipy.spatial import distance as dst
# import itertools
# import operator
# from scipy import stats
import time
from Moving_Pose_Descriptor import MP_tools2 as mpt
import Moving_Pose_Descriptor as mvp
import random
import math

start_time = time.time()
fv_subj = np.load('fv_subj.npy')

# fv_subj_less
#Build database with numerical labels

database = []

for iSubject in range(0, 12):
    for iAction in range(0, 11):
        for iframe in range(0, len(fv_subj[iSubject][iAction])):

            dt = tuple((fv_subj[iSubject][iAction][iframe], iSubject, iAction))
            database.append(dt)

database = np.array(database)

# Create Random Database
randframes = range(0,len(database))
random.shuffle(randframes)

#Percent of database
percent = 0.10 * len(database)

subrand = randframes[0:int(percent)]

testframes = database.copy() # The same database i pass it as na input

#compute confidence for every frame
k = 10 # k nearest neighbours
dist = []
class_frames = []
for income in range(0, testframes.shape[0]):

    # if testframes[income][3] != income:

        fv_in = testframes[income][0]
        print("incoming frame no:",income )

        for iframe in subrand:#range(0, database.shape[0]):
            # d = [dst.euclidean(fv_in, database[iframe][0]), database[iframe][1], database[iframe][2]]
            d = [dst.euclidean(fv_in, database[iframe][0]), database[iframe][1], database[iframe][2]]
            dist.append(d)

        # sort by distance
        dist.sort(key=mpt.getkey)

         #find most occurent class // most_common_occurence returns :(class, confidence )
        confidence_tuple = mpt.most_often_occurence(dist[0:k])
        cf = (fv_in, confidence_tuple)
        class_frames.append(cf)

class_frames = np.array(class_frames)
np.save('class_frames.npy',class_frames)

print("--- %s seconds ---" % (time.time() - start_time))

print()
