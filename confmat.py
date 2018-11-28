import os
from Moving_Pose_Descriptor import MP_tools2 as mpt
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from Moving_Pose_Descriptor.heatmap import heatmap
from Moving_Pose_Descriptor.heatmap import annotate_heatmap

def Conf2Subject(subject1,subject2,dtpath,fv_1,fv_2,params=None ):
    """Conf2Subj ::

    inputs: subject1,subject2,dataset path,feature vector of s1,FV of s2,
    params =[0 or 1,string] / 0: No plot saving ,1: saves all plots, string: contains the absolute path of the
    destination save folder.

    output: m x m score matrix between subj1,subj2 for act [0-11]
            class_score : (rowmin+colmin)/(2*num of actions) * %
            missclass"""
    act_s1, act_s1_not = mpt.list_ext(os.path.join(dtpath, subject1), 'json')
    act_s2, act_s2_not = mpt.list_ext(os.path.join(dtpath, subject2), 'json')

    act_s1_not = mpt.AlpNumSorter(act_s1_not)
    act_s2_not = mpt.AlpNumSorter(act_s2_not)
    print(act_s1_not)
    print(act_s2_not)
    print("new_pair:")
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

            # Scores of DTW for every subject / Objective Function
            score[sub1][sub2] = (C[-1, -1] / ((Y.shape[0] + Y.shape[1])))
            #
            mpt.DistMatPlot(Y, params[1], q, p, dtwscore=score[sub1][sub2],
                            name=act_s1_not[sub1] + "_" + act_s2_not[sub2], flag='DTW',
                            save_flag=params[0])

    #Class Score [min row + min col]
    # Pscore = ((score/np.amax(score))*100).copy()

    Pminrow = np.argmin(score, axis=1)  # axis=1 row min index
    Pmincol = np.argmin(score, axis=0)  # axis=0 col min index
    Pvec = np.arange(0, len(Pminrow))

    pr = Pminrow == Pvec
    pr = pr*1
    pc = Pmincol == Pvec
    pc = pc*1

    mtot = pr + pc

    missclass = (2*len(score)) - np.sum(mtot)
    # # print(missclass)
    class_score = (np.sum(mtot, dtype=float) / (2 * len(score))) * 100
    # print(class_score)

    return score, class_score , missclass

def cfm_savefig(subject1,subject2,params_cmf):
    """Saves Confusion matrix of 2 subjects and 11 actions at the specified path params_cmf[]
    input: subject1, subject2 names strings
    ouput: plt.savefig to path"""

    if params_cmf[4] == 1:

        goal_dir = os.path.join(params_cmf[5])

        axlabel = [subject2, subject1]  # [x,y]
        mpt.plot_confusion_matrix(params_cmf[0], classes=params_cmf[1], normalize=False, title='confusion matrix', axs=axlabel)
        # plt.plot(indexes[:,0], indexes[:,1],'ro')
        plt.title('Classification Score = ' + str(round(params_cmf[2], 2)) + '%\nmisclassified=' + str(int(params_cmf[3])))
        plt.rcParams.update({'font.size': 13})
        plt.tight_layout()
        # plt.show()

        name = subject1 + '_' + subject2
        my_file = name + '_cfm'

        plt.savefig(goal_dir + my_file)
        plt.close('all')

def avg_perf_savefig(avg_cscore, c_score, subj_name, params=None):
    """ Saves 2 figures at specified path using params key= list [0 or 1,save_path_string]
        -first element of the "params" list 1 or 0 activates or deactivates plot function.
        -second element is the string which specifies save path for the plot.
        fig1: Class scores 1 subj vs all subs
        fig2 : Average performance save fig """
    if params[0] == 1:

        goal_dir = os.path.join(params[1])

        # plot 1 subj vs all
        name1 = '1vsAll'
        axis1 = ['subjects', 'subjects']
        mpt.plot_confusion_matrix(c_score, classes=subj_name, normalize=False, title='mhad class score per subject', axs=axis1)
        plt.savefig(goal_dir + name1)
        plt.close('all')

        # # Plot Average Performance in dataset
        name2 = 'avg_performance_dtset'
        col_label = ['average']
        img_view = np.reshape(avg_cscore, (12, 1))
        im, cbar= heatmap(img_view, subj_name, col_label, cmap='jet')
        texts = annotate_heatmap(im, valfmt="{x:.2f} ")
        plt.axes().set_aspect('auto')
        plt.title('Average Performance for every Subject\n Mhad_dataset')
        plt.savefig(goal_dir + name2)
        plt.close('all')