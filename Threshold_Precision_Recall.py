import numpy as np
import os
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
from Moving_Pose_Descriptor import ThresPR as classify

# conf_1 = np.load('eval_mat.npy')
# conf = normalize(conf_1, norm='max')
# conf = conf_1 / np.amax(conf_1)

def precision_recall(conf_not):

    conf = conf_not / np.amax(conf_not)#_not normalized
    act_thres_pres_rec = np.zeros((11, 21, 2), dtype=float)
    # total =(tstep[0]/tstep[1]) + 1

    for label in range(0, 11):
        for t in range(0,21):
            thres = 0.05 * t

            tp = 0.0
            fp = 0.0
            tn = 0.0
            fn = 0.0

            for iSubject in range(0, 12):
                for iAction in range(0, 11):
                    answer = classify.belongsto(iSubject, iAction, label, thres, conf)

                    if (answer): # positive
                        if (iAction == label):
                            tp += 1
                        else:
                            fp += 1
                    else: # negative
                        if (iAction != label):
                            tn += 1
                        else:
                            fn += 1

                    precision = tp / (tp + fp + 0.001)
                    recall    = tp / (tp + fn + 0.001)

                    act_thres_pres_rec[label][t][0] = precision
                    act_thres_pres_rec[label][t][1] = recall

    # Plots
    actions = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11"]
    thresp = np.linspace(0,1,21)

    goal_dir = os.getcwd() + "/plots/conf_matrix/thres_prec_rec/"
    for iAction in range(0, 11):
        # create new graph
        recs = []
        precs = []
        for iThres in range(0, 21):
            # add act_thres_pres_rec[iAction][iThres][0/1] to graph
                r = [act_thres_pres_rec[iAction][iThres][1]]
                recs.extend(r)
                p = [act_thres_pres_rec[iAction][iThres][0]]
                precs.extend(p)
        plt.plot(thresp,recs,linestyle='-', marker='o', color='b',label='recall')
        plt.plot(thresp,precs,linestyle='-', marker='o', color='g',label='precision')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend(loc='upper right')
        plt.title("Action: " + actions[iAction] + "\nTHRES: 0:1:0.05")
        plt.savefig(goal_dir + actions[iAction])
        # plt.show()
        plt.close('all')

# print()
