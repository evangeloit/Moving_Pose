import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from Moving_Pose_Descriptor.heatmap import heatmap
from Moving_Pose_Descriptor.heatmap import annotate_heatmap
import math


def Conf2Subject(subject1, subject2, SubjectsActions, dtpath, fv_1, fv_2, params=None):

    act_s1, act_s1_not = mpt.list_ext(os.path.join(dtpath, subject1), 'json')
    act_s2, act_s2_not = mpt.list_ext(os.path.join(dtpath, subject2), 'json')

    act_s1_not = mpt.AlpNumSorter(act_s1_not)
    act_s2_not = mpt.AlpNumSorter(act_s2_not)

    act_s1_not = act_s1_not[0: SubjectsActions[1]]
    act_s2_not = act_s2_not[0: SubjectsActions[1]]


    score = np.empty((len(act_s1_not), len(act_s2_not)), np.dtype(np.float32))

    for act1 in range(0, len(act_s1_not)):
        for act2 in range(0, len(act_s2_not)):

            Y = cdist(fv_1[act1], fv_2[act2], 'euclidean')
            p, q, C, phi = mpt.dtwC(Y, 0.1)

            # Sum of diagonal steps
            # dsteps = 0
            # for r in range(0, len(q) - 1):
            #     qdot = abs(q[r] - q[r + 1])
            #     pdot = abs(p[r] - p[r + 1])
            #     s = qdot + pdot
            #     if s == 2:
            #         dsteps = dsteps + 1
                # print(dsteps)

            # Scores of DTW for every subject / Objective Function
            score[act1][act2] = (C[-1, -1] / ((Y.shape[0] + Y.shape[1])))

            mpt.DistMatPlot(Y, params[1], q, p, dtwscore=score[act1][act2],
                            name=act_s1_not[act1] + "_" + act_s2_not[act2], flag='DTW',
                            save_flag=params[0])

    Pminrow = np.argmin(score, axis=1)  # axis=1 row min index
    Pmincol = np.argmin(score, axis=0)  # axis=0 col min index
    Pvec = np.arange(0, len(Pminrow))

    pr = Pminrow == Pvec
    pr = pr*1
    pc = Pmincol == Pvec
    pc = pc*1

    mtot = pr + pc

    missclass = (2*len(score)) - np.sum(mtot)

    class_score = (np.sum(mtot, dtype=float) / (2 * len(score))) * 100

    return score, class_score, missclass

def cfm_savefig(subject1,subject2,params_cmf):
    """Saves Confusion matrix of 2 subjects and 11 actions at the specified path params_cmf[]
    input: subject1, subject2 names strings
    ouput: plt.savefig to path"""

    if params_cmf[4] == 1:

        goal_dir = os.path.join(params_cmf[5])

        axlabel = [subject2, subject1]  # [x,y]
        mpt.plot_confusion_matrix(params_cmf[0], classes=params_cmf[1], normalize=False, title='confusion matrix', axs=axlabel)
        plt.title('Classification Score = ' + str(round(params_cmf[2], 2)) + '%\nmisclassified=' + str(int(params_cmf[3])))
        plt.rcParams.update({'font.size': 13})
        plt.tight_layout()

        name = subject1 + '_' + subject2
        my_file = name + '_cfm'

        plt.savefig(goal_dir + my_file)

def evaluation_matrix(evmat, subjects, actions, savefig_eval=None):
    dims = subjects * actions
    new1 = np.zeros((dims, dims), dtype=float)

    for iRow in range(0, dims):
        for iCol in range(0, dims):
            iSub1 = iRow % subjects
            iAct1 = int(math.floor(iRow / subjects))

            iSub2 = iCol % subjects
            iAct2 = int(math.floor(iCol / subjects))
            new1[iRow][iCol] = evmat[iSub1][iSub2][iAct1][iAct2]

    if savefig_eval[0] == 1:
        figname = "Conf_Matrix_Total"
        fig1, ax1 = plt.subplots()
        ax1.imshow(new1, cmap='jet')
        ax1.set_title("Confusion Matrix\nDataset(" + str(subjects) + " Subjects x " + str(actions) + " Actions)")
        ax1.set_xlabel("Actions")
        ax1.set_ylabel("Actions")
        ax1.set_aspect(aspect='equal', adjustable='box')
        fig1.tight_layout()
        fig1.savefig(savefig_eval[1] + figname)

    return new1