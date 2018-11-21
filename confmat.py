import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from scipy.spatial.distance import cdist
import numpy as np

def Conf2Subject(subject1,subject2,dtpath,fv_1,fv_2,params=None ):

    act_s1, act_s1_not = mpt.list_ext(os.path.join(dtpath, subject1), 'json')
    act_s2, act_s2_not = mpt.list_ext(os.path.join(dtpath, subject2), 'json')

    act_s1_not = mpt.AlpNumSorter(act_s1_not)
    act_s2_not = mpt.AlpNumSorter(act_s2_not)
    print(act_s1_not)
    print(act_s2_not)
    score = np.empty((len(act_s1_not), len(act_s2_not)), np.dtype(np.float32))

    for sub1 in range(0, len(act_s1_not)):
        for sub2 in range(0, len(act_s2_not)):

            Y = cdist(fv_1[sub1], fv_2[sub2], 'euclidean')
            p, q, C, phi = mpt.dtwC(Y, 0.1)

            # Sum of diagonal steps
            dsteps = 0
            for r in range(0, len(q) - 1):
                qdot = abs(q[r] - q[r + 1])
                pdot = abs(p[r] - p[r + 1])
                s = qdot + pdot
                if s == 2:
                    dsteps = dsteps + 1
                # print(dsteps)

            # Scores of DTW for every subject
            score[sub1][sub2] = (C[-1, -1] / ((Y.shape[0] + Y.shape[1])))

            mpt.DistMatPlot(Y, params[1], q, p, dtwscore=score[sub1][sub2],
                            name=act_s1_not[sub1] + "_" + act_s2_not[sub2], flag='DTW',
                            save_flag=params[0])

    return score