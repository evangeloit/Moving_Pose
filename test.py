import numpy as np
import time
from Moving_Pose_Descriptor import FrameWiseClassify as FrameWiseClassify
from Moving_Pose_Descriptor import MP_tools2 as mpt
import random

start_time = time.time()
fv_subj = np.load('fv_subj.npy')

classified =np.load('class_frames_k5.npy')
# Build database with numerical labels

database = []

for iSubject in range(0, 3):
    for iAction in range(0, 4):
        for iframe in range(0, len(fv_subj[iSubject][iAction])):

            dt = tuple((fv_subj[iSubject][iAction][iframe], iSubject, iAction))
            database.append(dt)


database = np.array(database)

testframes = database.copy()

# compute confidence for every frame
k = 4
class_frames = []
ct = 0

for income in range(0, 1100):

        #incoming frame Feature Vector
        fv_in = testframes[income][0]
        # print("incoming frame no:",income ,"  iSub", testframes[income][1], "  iAct", testframes[income][2] )
        print("frame :", income)#,"  iSub", testframes[income][1], "  iAct", testframes[income][2] )

        #filter Database
        newdatabase = mpt.filterdatabase(testframes[income],database) #exclude iSubject's all iAction frames

        #Random frame numbers
        partData=None#mpt.randomData(newdatabase, percent=None)

        #confidence of frame that belong to a class (frame_feature vector ,
        confidence_tuple = FrameWiseClassify.classframe(fv_in, newdatabase, k, dataPercent=partData)

        change_action = abs(testframes[income][2] - testframes[income+1][2])

        if change_action != 0:
            ct = 0

        ct = ct + 1

        cf = (ct, confidence_tuple)
        class_frames.append(cf)

class_frames = np.array(class_frames, dtype=object)
np.save('class_frames_k5.npy',class_frames)

print("--- %s seconds ---" % (time.time() - start_time))

print()
