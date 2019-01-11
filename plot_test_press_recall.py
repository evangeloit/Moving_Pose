import numpy as np
import os
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
from Moving_Pose_Descriptor import ThresPR as classify


#k = mostConf[(np.where((mostConf[:,1] == iSubject) & (mostConf[:,2]==iAction)))]

act_thres_pres_rec = np.load('act_thres_pres_rec.npy')

# zeros = k[k != 0]
actions = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11"]
thresp = np.linspace(0,1,21)

# TODO: put the directory out of the function
goal_dir = os.getcwd() + "/plots/conf_matrix/thres_prec_rec/"
for iAction in range(0, 11):
    # create new graph
    recs = []
    precs = []
    fp = []
    acc = []
    for iThres in range(0, 21):
        # add act_thres_pres_rec[iAction][iThres][0/1] to graph
            r = [act_thres_pres_rec[iAction][iThres][1]]
            recs.extend(r)
            p = [act_thres_pres_rec[iAction][iThres][0]]
            precs.extend(p)
            f = [act_thres_pres_rec[iAction][iThres][2]]
            fp.extend(f)
            ac = [act_thres_pres_rec[iAction][iThres][3]]
            acc.extend(ac)

    plt.plot(thresp[4:], recs[4:], linestyle='-', marker='o', color='b', label='recall')
    plt.plot(thresp[4:], precs[4:], linestyle='-', marker='o', color='g', label='precision')
    # plt.plot(fp, recs, linestyle='--', marker='o', color='r', label='ROC')
    plt.plot(thresp[4:], acc[4:], linestyle='-', marker='*', color='m', label='Accuracy')

    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend(loc='upper right')
    plt.title("Action: " + actions[iAction] + "\nTHRES: 0:1:0.05")
    plt.savefig(goal_dir + actions[iAction])
    # plt.show()
    plt.close('all')