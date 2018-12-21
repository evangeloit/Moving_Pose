import numpy as np
import time
from Moving_Pose_Descriptor import FrameWiseClassify as FrameWiseClassify
from Moving_Pose_Descriptor import MP_tools2 as mpt
import random

start_time = time.time()
fv_subj = np.load('fv_subj.npy')

# classified =np.load('class_frames_k5.npy')
# Build database with numerical labels
print(fv_subj.size)
# database = []
database = np.zeros((3,4,100),dtype=object)
for iSubject in range(0, 3):
    for iAction in range(0, 4):
        for iframe in range(0, len(fv_subj[iSubject][iAction])):

        # for iframe in range(0, len(fv_subj[iSubject][iAction])):

            dt = [fv_subj[iSubject][iAction][iframe], iSubject, iAction]
            database[iSubject][iAction][iframe]= dt
            # database.append(dt)

database = np.array(database)


print()
