import numpy as np
import time
from Moving_Pose_Descriptor import FrameWiseClassify as FrameWiseClassify
import random

start_time = time.time()
fv_subj = np.load('fv_subj.npy')

class_k5 = np.load('class_frames_k5.npy')

# Build database with numerical labels

database = []

for iSubject in range(0, 12):
    for iAction in range(0, 11):
        for iframe in range(0, len(fv_subj[iSubject][iAction])):

            dt = tuple((fv_subj[iSubject][iAction][iframe], iSubject, iAction))
            database.append(dt)

# filter_database(isubject, database)
database = np.array(database)

# create random frame numbers based on database and keep a % of them
randpicks = range(0, len(database))
random.shuffle(randpicks)
percent = 0.10 * len(randpicks)
partData = randpicks[0:int(percent)]


testframes = database.copy() # The same database i pass it as na input

# compute confidence for every frame
k = 5
class_frames = []
for income in range(0, testframes.shape[0]):

        #incoming frame Feature Vector
        fv_in = testframes[income][0]
        # print("incoming frame no:",income ,"  iSub", testframes[income][1], "  iAct", testframes[income][2] )
        # print "ok"
        #confidence of frame that belong to a class (frame_feature vector ,
        confidence_tuple = FrameWiseClassify.classframe(fv_in, database, k, dataPercent= partData)

        cf = (fv_in, confidence_tuple)
        class_frames.append(cf)

class_frames = np.array(class_frames)
np.save('class_frames_k5.npy',class_frames)

print("--- %s seconds ---" % (time.time() - start_time))

print()
