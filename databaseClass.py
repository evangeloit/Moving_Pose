#Reduced database / randomly picked percentage of the input database

# import numpy as np
# import time
# from Moving_Pose_Descriptor import FrameWiseClassify as FrameWiseClassify
# from Moving_Pose_Descriptor import MP_tools2 as mpt
import random
#
# start_time = time.time()
# fv_subj = np.load('fv_subj.npy')
#
# # Build database with numerical labels
# database = []
#
# for iSubject in range(0, 12):
#     for iAction in range(0, 11):
#         for iframe in range(0, len(fv_subj[iSubject][iAction])):
#
#             dt = tuple((fv_subj[iSubject][iAction][iframe], iSubject, iAction))
#             database.append(dt)
#
#
# database = np.array(database)




# class reduceDatabase:
#
#
#     def __init__(self,database,percent):
#         self.database = database
#         self.percent = percent
#
#     def randDataIndex(self):
#
#         randpicks = range(0, len(self.database))
#         random.shuffle(randpicks)
#         percent = self.percent * len(randpicks)
#         partData = randpicks[0:int(percent)]
#
#         return partData

# @fullDatabase = array[N][[features...], subject, action]
def reduceDatabase(fullDatabase, trainCount, testCount):
    indices = range(0, len(fullDatabase))
    random.shuffle(indices)

    return (
        fullDatabase[indices[0 : trainCount]],
        fullDatabase[indices[trainCount : trainCount + testCount]]
    )
#
# #Test
# percent_set = 0.1
# percent_testset = 0.05
# r_database = reduceDatabase(database, percent_set)
# r_testset = reduceDatabase(database, percent_testset)
#
# DATABASE = database[r_database.randDataIndex()]
# TESTSET = database[r_testset.randDataIndex()]
# # print(r_database.randDataIndex())
# # print(r_testset.randDataIndex())

# print(r_testset.percent)
# print()

