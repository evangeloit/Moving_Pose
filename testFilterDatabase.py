import numpy as np
import time
from Moving_Pose_Descriptor import FrameWiseClassify as FrameWiseClassify
from Moving_Pose_Descriptor import MP_tools2 as mpt
import random

start_time = time.time()
fv_subj = np.load('fv_subj.npy')

# classified =np.load('class_frames_k5.npy')
# Build database with numerical labels

database = []

for iSubject in range(0, 12):
    for iAction in range(0, 11):
        for iframe in range(0, len(fv_subj[iSubject][iAction])):

            dt = tuple((fv_subj[iSubject][iAction][iframe], iSubject, iAction))
            database.append(dt)


database = np.array(database)

testframes = database.copy()

# compute confidence for every frame
k = 5
class_frames = []
for income in range(0, testframes.shape[0]):

        #incoming frame Feature Vector
        fv_in = testframes[income][0]
        # print("incoming frame no:",income ,"  iSub", testframes[income][1], "  iAct", testframes[income][2] )
        print("frame :", income)#,"  iSub", testframes[income][1], "  iAct", testframes[income][2] )

        #filter Database
        newdatabase = mpt.filterdatabase(testframes[income],database) #exclude iSubject's all iAction frames

        #Random frame numbers
        partData=mpt.randomData(newdatabase, percent=0.1)

        #confidence of frame that belong to a class (frame_feature vector ,
        confidence_tuple = FrameWiseClassify.classframe(fv_in, newdatabase, k, dataPercent=partData)

        cf = (fv_in, confidence_tuple)
        class_frames.append(cf)

class_frames = np.array(class_frames)
np.save('class_frames_k5.npy',class_frames)

print("--- %s seconds ---" % (time.time() - start_time))

print()
